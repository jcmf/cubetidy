// End-to-end check of the line detector against KNOWN ground truth — the offline
// test the repo otherwise can't do. Renders synthetic cubes (tools/synth-cube.mjs)
// at chosen poses, pushes each through the REAL pixel pipeline (detectLineSegments =
// Canny + HoughLinesP, then solveCubeFromLines), and grades the recovered pose
// against truth: it must LOCK, its rotation must match (mod the 24 cube symmetries,
// via canonicalizeRotation), and its projected centre + depth must match.
//
// This proves the whole stack — pixels in, metric pose out — not just the geometry
// half that test/lines.test.mjs covers from pre-projected segments. Like
// tools/detect-smoke.mjs it loads the ~10 MB OpenCV WASM, so it is NOT part of
// `npm test`; run it manually:  node tools/synth-smoke.mjs
//
// A FAILURE here means either the detector regressed or the synthetic render drifted
// from the model — both worth knowing. Tune knobs per-scene via the SCENES table.

import { Canvas } from 'skia-canvas';
import cv from '@techstark/opencv-js';
import { buildCubeScene, renderScene } from './synth-cube.mjs';
import { detectAndSolveLines } from '../src/detect.js';
import { canonicalizeRotation, rotationAngleDeg } from '../src/lines.js';
import { project } from '../src/pose.js';
import { readStickerColors, nearestFaceLetter } from '../src/read-colors.js';

await new Promise((r) => { if (cv && cv.Mat) return r(); cv.onRuntimeInitialized = r; });

// Poses chosen to span the detector's regime: strong perspective (close), three
// corner-on faces with separated VPs. Far/near-affine views are deliberately out of
// scope (the detector itself doesn't claim them — see lines.js step-2 KEY LIMIT).
// The dist=3 entries pin the VERY-close / strong-perspective end (cube near or past the
// frame edge): there the coverage search's coarse lateral step must stay ≤ tol or it
// skips the true peak and the lock lands one cell off — guard against that regressing.
const SCENES = [
  { tag: 'corner-on A close', axis: '0.9,-1,0.1', angleDeg: 57, dist: 4.5 },
  { tag: 'corner-on B close', axis: '0.6,-1,0.2', angleDeg: 60, dist: 4.5 },
  { tag: 'corner-on A mid', axis: '0.9,-1,0.1', angleDeg: 57, dist: 6 },
  { tag: 'tilted', axis: '0.4,-1,0.5', angleDeg: 65, dist: 5 },
  { tag: 'corner-on dist3', axis: '-0.55,-1,0.1', angleDeg: 57, dist: 3, tx: 0.05, ty: -0.1 },
  { tag: 'tilted dist3', axis: '0.4,-1,0.5', angleDeg: 65, dist: 3.2, tx: -0.1, ty: 0.05 },
  // Near-EDGE-ON: only 2 faces visible. A close-to-affine starved VP family leaves the
  // lateral cell under-constrained; guards the cell-hop translation fix that lands it on
  // the right sticker (the orthographic 3-face model + single search put it a cell off).
  { tag: 'edge-on 2-face', axis: '0.15,-1,0.1', angleDeg: 57, dist: 3, tx: 0.05, ty: -0.2 },
  // Per-sticker colours: the colour-read check below is only conclusive here (on a
  // solved cube a one-cell slip still reads the face's own colour).
  { tag: 'corner-on A scrambled', axis: '0.9,-1,0.1', angleDeg: 57, dist: 4.5, scramble: 3 },
];

const ANGLE_TOL = 8;     // recovered R within this many degrees of truth (mod symmetry)
const CENTRE_TOL = 0.25; // projected centre within this fraction of the cube edge
const DEPTH_TOL = 0.18;  // recovered depth within this fraction of true depth

let pass = 0, fail = 0;
const check = (name, cond, extra = '') => { cond ? pass++ : fail++; console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`)); return cond; };

for (const spec of SCENES) {
  const scene = buildCubeScene(spec);
  const canvas = renderScene(scene, { gap: 0.1, bg: '#15151a' });
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, scene.width, scene.height);

  // The worker-identical entry point (includes the soft-frame Canny retry).
  const { segments, sol, retried } = detectAndSolveLines(cv, imageData, scene.K, {});
  const fit = sol && sol.fit;

  console.log(`\n[${spec.tag}] ${segments.length} segments${retried ? ' (retry)' : ''} | ${fit ? `${fit.locked ? 'LOCK' : 'unlocked'} ${fit.count}pts cover ${(fit.cover || 0).toFixed(2)} reproj ${fit.reprojErr?.toFixed?.(1)}px` : 'no fit'}`);

  if (!check(`[${spec.tag}] locks`, !!(fit && fit.locked), 'no lock')) continue;

  // Rotation: align the recovered R to the symmetry rep nearest truth before comparing.
  const Rc = canonicalizeRotation(fit.pose.R, scene.pose.R);
  const ang = rotationAngleDeg(scene.pose.R, Rc);
  check(`[${spec.tag}] rotation within ${ANGLE_TOL}°`, ang <= ANGLE_TOL, `${ang.toFixed(1)}°`);

  // Translation: the cube CENTRE is symmetry-invariant, so compare it directly.
  const cTrue = project(scene.K, scene.pose, [0, 0, 0]);
  const cFit = project(scene.K, fit.pose, [0, 0, 0]);
  const centreErr = Math.hypot(cFit[0] - cTrue[0], cFit[1] - cTrue[1]) / scene.truth.edgePx;
  check(`[${spec.tag}] centre within ${CENTRE_TOL}·edge`, centreErr <= CENTRE_TOL, `${centreErr.toFixed(2)}·edge`);

  const depthErr = Math.abs(fit.pose.t[2] - scene.pose.t[2]) / scene.pose.t[2];
  check(`[${spec.tag}] depth within ${(DEPTH_TOL * 100) | 0}%`, depthErr <= DEPTH_TOL, `${(depthErr * 100).toFixed(0)}%`);

  // Colour read under the recovered (truth-aligned) pose. Graded over the cells the
  // reader CLAIMS (weight > 0): a claimed-but-wrong colour is the real failure; a
  // weight-0 cell (grazing face, off-frame) is a declared no-read — the live
  // accumulator just waits for a better view of it. Claimed must still cover at
  // least two full faces so the check can't pass by claiming nothing.
  const reads = readStickerColors(imageData, scene.K, { R: Rc, t: fit.pose.t });
  const byFace = new Map(reads.faces.map((f) => [`${f.k}${f.s}`, f]));
  let colOk = 0, claimed = 0, colN = 0;
  for (const g of scene.truth.facelets) {
    const rf = byFace.get(`${g.k}${g.s}`);
    g.cells.forEach((row, r) => row.forEach((cell, c) => {
      colN++;
      const rc = rf && rf.cells[r * 3 + c];
      if (!rc || !(rc.weight > 0)) return;
      claimed++;
      if (nearestFaceLetter(rc.rgb) === cell.letter) colOk++;
    }));
  }
  check(`[${spec.tag}] colours ${colOk}/${claimed} claimed (of ${colN})`,
    colOk === claimed && claimed >= 18, `${colOk}/${claimed}`);
}

console.log(`\n${fail ? 'FAILED' : 'OK'} — ${pass} passed, ${fail} failure(s)`);
process.exit(fail ? 1 : 0);

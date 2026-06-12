// Colour reader (read-colors.js) against REAL rendered pixels — the one slice of
// the pixel-sampling layer `npm test` can cover. drawScene needs only
// createImageData/putImageData, so a stub context rasterizes a synthetic cube in
// plain node (no skia, no browser), and sampling at the scene's TRUTH pose needs no
// detector (no OpenCV WASM). What this proves: the reader's face/cell indexing
// matches buildCubeScene's truth facelets index-for-index (the scrambled case
// catches any row/col transposition or mirror on a negative-axis face), the patch
// median survives blur+noise, and the confidence signals behave. What it can't
// prove: reads under a RECOVERED pose (tools/synth-bench.mjs grades that) or real
// lighting (a physical cube in the browser).
//
// Run: npm test

import { buildCubeScene, drawScene } from '../src/synth.js';
import {
  readStickerColors, nearestFaceLetter,
  accumulateStickerColors, accumulatedColors, overlayColors,
} from '../src/read-colors.js';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`));
}

// Render a scene through a stub context and return the ImageData drawScene stored.
function renderImage(scene, opts) {
  let stored = null;
  const ctx = {
    createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (img) => { stored = img; },
  };
  drawScene(ctx, scene, opts);
  return stored;
}

const readAt = (sceneOpts, readOpts = {}) => {
  const scene = buildCubeScene(sceneOpts);
  const read = readStickerColors(renderImage(scene, sceneOpts), scene.K, scene.pose, readOpts);
  return { scene, read };
};

// Truth letters for the reader's face (k, s), flattened row-major like its cells.
const truthFace = (scene, f) => scene.truth.facelets.find((g) => g.k === f.k && g.s === f.s);
const truthLetters = (g) => g.cells.flatMap((row) => row.map((c) => c.letter));

// Count reader cells whose nearest-palette letter matches truth, across all faces.
function gradeRead(scene, read) {
  let okC = 0, n = 0;
  for (const f of read.faces) {
    const letters = truthLetters(truthFace(scene, f));
    f.cells.forEach((c, i) => { n++; if (c.rgb && nearestFaceLetter(c.rgb) === letters[i]) okC++; });
  }
  return { okC, n };
}

// --- solved cube, default pose: geometry + colours line up with truth ---------
{
  const { scene, read } = readAt({});
  check('default scene: three faces visible', read.faces.length === 3);
  check('faces match the truth visible set',
    read.faces.every((f) => scene.truth.visibleFaces.some((v) => v.k === f.k && v.s === f.s)));

  // Cell centres must reproject exactly onto the truth facelets' centres — this
  // pins the row-major (r, c) ↔ in-plane-axes convention to buildCubeScene's.
  let maxD = 0;
  for (const f of read.faces) {
    const g = truthFace(scene, f);
    f.cells.forEach((c, i) => {
      const t = g.cells[(i / 3) | 0][i % 3].centre2D;
      maxD = Math.max(maxD, Math.hypot(c.px[0] - t[0], c.px[1] - t[1]));
    });
  }
  check('cell centres match truth centres (indexing)', maxD < 1e-6, `max ${maxD}px`);

  const { okC, n } = gradeRead(scene, read);
  check(`solved: all ${n} cells classify to their face colour`, okC === n, `${okC}/${n}`);
  check('all cells carry weight', read.faces.every((f) => f.cells.every((c) => c.weight > 0)));
}

// --- scrambled: per-sticker colours catch any index transposition/mirror ------
{
  const { scene, read } = readAt({ scramble: 5 });
  const { okC, n } = gradeRead(scene, read);
  check(`scrambled: all ${n} cells match their own truth sticker`, okC === n, `${okC}/${n}`);
}

// --- a mediocre webcam: blur + noise, still scrambled --------------------------
{
  const { scene, read } = readAt({ scramble: 7, imgBlur: 2, noise: 10, seed: 11 });
  const { okC, n } = gradeRead(scene, read);
  check(`blur+noise: ≥${n - 1}/${n} cells still classify correctly`, okC >= n - 1, `${okC}/${n}`);
}

// --- near-edge-on pose: only the genuinely visible 2 faces are read -----------
{
  const { read } = readAt({ axis: '0.15,-1,0.1', angleDeg: 57, dist: 3, tx: 0.05, ty: -0.2 });
  check('near-edge-on: two faces read', read.faces.length === 2);
}

// --- cube partly out of frame: flagged via inFrame/weight, no crash -----------
{
  const { read } = readAt({ dist: 3, tx: 1.2 });
  const cells = read.faces.flatMap((f) => f.cells);
  check('off-frame: some cells lose patch points', cells.some((c) => c.inFrame < 1));
  check('off-frame: rgb-less cells carry zero weight', cells.every((c) => c.rgb || c.weight === 0));
}

// --- accumulation: weighted mean, stable keys, low weight barely moves it -----
{
  const a = readAt({ scramble: 5, noise: 8, seed: 3 });
  const b = readAt({ scramble: 5, noise: 8, seed: 4 }); // same cube, different noise field
  let st = accumulateStickerColors(null, a.read);
  st = accumulateStickerColors(st, b.read);
  const acc = accumulatedColors(st);
  check('accumulate: every cell saw both reads',
    acc.faces.every((f) => f.cells.every((c) => c.n === 2)));
  let okC = 0, n = 0;
  for (const f of acc.faces) {
    const letters = truthLetters(truthFace(a.scene, f));
    f.cells.forEach((c, i) => { n++; if (c.rgb && nearestFaceLetter(c.rgb) === letters[i]) okC++; });
  }
  check(`accumulate: all ${n} accumulated cells classify correctly`, okC === n, `${okC}/${n}`);

  // A near-zero-weight garbage read must not displace an established mean.
  const garbage = {
    faces: a.read.faces.map((f) => ({
      ...f,
      cells: f.cells.map((c) => ({ ...c, rgb: { r: 255, g: 0, b: 255 }, weight: 1e-4 })),
    })),
  };
  const st2 = accumulateStickerColors(st, garbage);
  const c0 = accumulatedColors(st).faces[0].cells[0].rgb;
  const c2 = accumulatedColors(st2).faces[0].cells[0].rgb;
  const drift = Math.hypot(c2.r - c0.r, c2.g - c0.g, c2.b - c0.b);
  check('accumulate: low-weight garbage barely moves the mean', drift < 1, `${drift.toFixed(2)}`);

  // overlayColors: current read's positions, accumulator's colours.
  const ov = overlayColors(b.read, st);
  check('overlay: keeps the current read\'s pixel positions',
    ov[0].cells[0].px === b.read.faces[0].cells[0].px);
  const accRgb = accumulatedColors(st).faces.find((f) => f.k === ov[0].k && f.s === ov[0].s).cells[0].rgb;
  check('overlay: shows the accumulated colour', ov[0].cells[0].rgb.r === accRgb.r);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

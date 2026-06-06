// Verifies src/cube-pose.js: from a synthetic corner-on cube projected to sticker
// quads, the detect->group->fit->global-pose chain recovers ONE cube pose that
// reprojects the whole cube (all visible faces, including stickers not used to seed
// it) back onto the truth. Offline; the live stability win can't be unit-tested,
// but pose determinism + correctness can.
//
// Run: npm test

import { estimateIntrinsics } from '../src/pose.js';
import { projectCube, estimateCubePose } from '../src/cube-pose.js';
import { findFaceGrids, fitFaceGrid } from '../src/group.js';

let pass = 0, fail = 0;
const check = (name, cond, extra = '') => { cond ? pass++ : fail++; console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`)); };

const col = (R, j) => [R[0][j], R[1][j], R[2][j]];
const mul = (A, B) => A.map((r, i) => B[0].map((_, j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const rotX = (a) => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
const rotY = (a) => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];

const K = estimateIntrinsics(1280, 720, 60);

function quadAt(x, y, area) {
  const s = Math.sqrt(area) / 2;
  return { center: { x, y }, area, corners: [{ x: x - s, y: y - s }, { x: x + s, y: y - s }, { x: x + s, y: y + s }, { x: x - s, y: y + s }] };
}

// Build a corner-on truth cube, project its visible stickers to quads, recover.
function runPose(yaw, pitch, C, label) {
  const R = mul(rotY(yaw), rotX(pitch));
  const visible = projectCube(R, C, K);
  check(`${label}: corner-on shows 3 faces`, visible.length === 3, `got ${visible.length}`);

  // Projected sticker centres -> quads sized to the local pitch.
  const centers = visible.flatMap((f) => f.cells);
  const nn = centers.map((p, i) => Math.min(...centers.filter((_, j) => j !== i).map((q) => Math.hypot(p.x - q.x, p.y - q.y))));
  const pitchPx = [...nn].sort((a, b) => a - b)[nn.length >> 1];
  const quads = centers.map((p) => quadAt(p.x, p.y, (pitchPx * 0.8) ** 2));

  const fits = findFaceGrids(quads).map((f) => fitFaceGrid(f)).filter(Boolean);
  check(`${label}: groups at least one face`, fits.length >= 1, `got ${fits.length}`);

  const est = estimateCubePose(fits, K, quads);
  check(`${label}: estimates a cube pose`, !!est);

  // Every truth visible sticker should have a recovered sticker on top of it.
  const estPts = est.faces.flatMap((f) => f.cells);
  let worst = 0;
  for (const t of centers) worst = Math.max(worst, Math.min(...estPts.map((p) => Math.hypot(p.x - t.x, p.y - t.y))));
  check(`${label}: recovered cube reprojects onto every visible sticker`, worst < 2, `worst=${worst.toFixed(2)}px`);
  check(`${label}: recovers all three visible faces`, est.faces.length === 3, `got ${est.faces.length}`);
}

runPose(0.62, 0.62, [0.4, -0.2, 11], 'pose A');
runPose(-0.7, 0.55, [-0.6, 0.3, 12], 'pose B');

// Determinism / stability: tiny perturbations of the inputs must give nearly the
// same pose (the whole point — near-identical frames -> near-identical pose).
(() => {
  const R = mul(rotY(0.6), rotX(0.6));
  const centers = projectCube(R, [0.3, -0.1, 11], K).flatMap((f) => f.cells);
  const nn = centers.map((p, i) => Math.min(...centers.filter((_, j) => j !== i).map((q) => Math.hypot(p.x - q.x, p.y - q.y))));
  const pitchPx = [...nn].sort((a, b) => a - b)[nn.length >> 1];
  const jitter = (seed) => { let s = seed; const r = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff - 0.5) * 2; }; return r; };

  // What the user sees is the 2D overlay, so measure REPROJECTION spread (depth may
  // wobble harmlessly). For each trial, the projected sticker positions; compare
  // each trial's points to trial 0's by nearest match.
  const trials = [];
  for (let trial = 0; trial < 4; trial++) {
    const r = jitter(trial + 1);
    const quads = centers.map((p) => quadAt(p.x + r(), p.y + r(), (pitchPx * 0.8) ** 2)); // +-1px noise
    const fits = findFaceGrids(quads).map((f) => fitFaceGrid(f)).filter(Boolean);
    const est = estimateCubePose(fits, K, quads);
    trials.push(est.faces.flatMap((f) => f.cells));
  }
  let spread = 0;
  for (let t = 1; t < trials.length; t++)
    for (const p of trials[0]) spread = Math.max(spread, Math.min(...trials[t].map((q) => Math.hypot(p.x - q.x, p.y - q.y))));
  check('stability: ±1px input noise keeps the overlay within a few px', spread < 6, `reproj spread=${spread.toFixed(2)}px`);
})();

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

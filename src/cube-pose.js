// Global cube-pose estimation: recover ONE rigid 6-DoF cube pose from the detected
// face grids, so the (up to) three visible faces come from a single hypothesis
// rather than independent per-face fits that fight and flicker frame to frame.
//
// 1. Seed: one face fully determines the cube. Its homography gives the face's
//    plane pose (poseFromHomography); its in-plane axes and normal ARE the cube's
//    axes (sticker pitch = 1 grid unit) and the centre sits 1.5 units behind it.
// 2. Score: project the whole seeded cube and count detected quads it explains;
//    keep the best seed. Correct faces agree on the same cube; spurious score low.
// 3. Refine: a single small face fixes orientation but not depth well, so fuse ALL
//    matched stickers (2-3 faces) with refinePnP — that's what makes the pose
//    stable frame to frame (the headline fix for corner-on jitter).
//
// Pure JS, offline-tested in test/cube-pose.test.mjs.

import { poseFromHomography, refinePnP } from './pose.js';

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const det3 = (a, b, c) =>
  a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
const col = (R, j) => [R[0][j], R[1][j], R[2][j]];
const apply = (R, t, p) => [
  R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
  R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
  R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
];
const rot = (R, p) => [
  R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
  R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
  R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
];

const HALF = 1.5; // cube half-size in sticker-pitch units (3 stickers across)

// Canonical cube model (origin-centred, axis-aligned): 6 faces, each its outward
// normal and 9 sticker centres in model space. Built once.
const MODEL = (() => {
  const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const faces = [];
  for (let a = 0; a < 3; a++) {
    for (const s of [1, -1]) {
      const N = scale(ax[a], s), A = ax[(a + 1) % 3], B = ax[(a + 2) % 3];
      const center = scale(N, HALF);
      const cells = [];
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) cells.push(add(add(center, scale(A, i)), scale(B, j)));
      faces.push({ normal: N, cells });
    }
  }
  return faces;
})();

const projPt = (K, P) => [K.f * P[0] / P[2] + K.cx, K.f * P[1] / P[2] + K.cy];

// Seed a cube pose { R, t } from one face fit. R = [u v n] (camera-space cube
// axes), t = cube centre; n is the seed face's outward normal (toward the camera).
export function seedFromFace(fit, K) {
  const fp = poseFromHomography(fit.H, K); // R,t for model points (col, row, 0)
  const fc = add(add(col(fp.R, 0), col(fp.R, 1)), fp.t); // seed face centre = R*(1,1,0)+t
  let u = col(fp.R, 0), v = col(fp.R, 1), n = col(fp.R, 2);
  if (dot(n, fc) > 0) n = scale(n, -1);    // normal toward the camera
  if (det3(u, v, n) < 0) u = scale(u, -1); // right-handed (non-mirrored) cube
  const C = sub(fc, scale(n, HALF));       // centre behind the visible face
  return { R: [[u[0], v[0], n[0]], [u[1], v[1], n[1]], [u[2], v[2], n[2]]], t: C };
}

// Project the front-facing faces of a posed cube: [{ normal, cells:[{x,y}*9] }].
export function projectCube(R, t, K) {
  const out = [];
  for (const f of MODEL) {
    const center = apply(R, t, scale(f.normal, HALF));
    if (dot(rot(R, f.normal), center) >= 0) continue;     // back-facing
    const pts = f.cells.map((p) => apply(R, t, p));
    if (pts.some((P) => P[2] <= 0)) continue;             // behind camera
    out.push({ normal: f.normal, cells: pts.map((P) => { const [x, y] = projPt(K, P); return { x, y }; }) });
  }
  return out;
}

// Match each detected quad to the nearest projected visible sticker (within tol),
// returning { X (model 3D), u (image) } correspondences for refinement.
function correspondences(R, t, K, quads, tol) {
  const proj = [];
  for (const f of MODEL) {
    const center = apply(R, t, scale(f.normal, HALF));
    if (dot(rot(R, f.normal), center) >= 0) continue;
    for (const p of f.cells) {
      const P = apply(R, t, p);
      if (P[2] > 0) { const [x, y] = projPt(K, P); proj.push({ X: p, x, y }); }
    }
  }
  const corr = [];
  for (const q of quads) {
    let best = null, bd = tol;
    for (const pr of proj) {
      const d = Math.hypot(pr.x - q.center.x, pr.y - q.center.y);
      if (d < bd) { bd = d; best = pr; }
    }
    if (best) corr.push({ X: best.X, u: [q.center.x, q.center.y] });
  }
  return corr;
}

// Estimate the cube pose: seed from each face fit, score by reprojection against
// all detected quads, refine the best with all its inliers. Returns
// { R, t, score, faces: projectCube(...) } or null.
export function estimateCubePose(fits, K, quads, opts = {}) {
  if (!fits.length) return null;
  const ref = fits.reduce((a, b) => (b.cells.length > a.cells.length ? b : a));
  const pitch = Math.hypot(ref.outline[1].x - ref.outline[0].x, ref.outline[1].y - ref.outline[0].y) / 3;
  const tol = (opts.matchFrac ?? 0.5) * pitch;

  let best = null;
  for (const fit of fits) {
    const { R, t } = seedFromFace(fit, K);
    const score = correspondences(R, t, K, quads, tol).length;
    if (!best || score > best.score) best = { R, t, score };
  }

  // Refine the winning seed over every sticker it explains, re-selecting inliers
  // between passes (RANSAC-lite) so outlier quads are dropped and the pose tightens.
  let { R, t } = best;
  for (let passNo = 0; passNo < 2; passNo++) {
    const corr = correspondences(R, t, K, quads, tol);
    if (corr.length < 6) break;
    ({ R, t } = refinePnP(R, t, corr.map((c) => c.X), corr.map((c) => c.u), K));
  }
  const score = correspondences(R, t, K, quads, tol).length;
  return { R, t, score, faces: projectCube(R, t, K) };
}

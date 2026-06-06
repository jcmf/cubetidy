// Global cube-pose estimation: recover ONE rigid 6-DoF cube pose from the detected
// face grids, so the (up to) three visible faces come from a single hypothesis
// rather than independent per-face fits that fight and flicker frame to frame.
//
// 1. Seed: one face fully determines the cube. Its homography gives the face's
//    plane pose (poseFromHomography); its in-plane axes and normal ARE the cube's
//    axes (sticker pitch = 1 grid unit) and the centre sits 1.5 units behind it.
// 2. Score (shape-aware): a detected sticker counts only if a projected sticker
//    lands on it AND their four CORNERS agree — a wrong pose can match centres but
//    not sticker shapes, so this separates real from false (lets minScore filter).
// 3. Refine: fuse the four CORNERS of every matched sticker (not just centres) with
//    refinePnP. The corners encode foreshortening, so the pose must agree with the
//    sticker shapes detection actually saw — the key to a stable, correct corner-on
//    pose. RANSAC-lite inlier re-selection between passes drops outlier quads.
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

// Canonical cube model: 6 faces, each its outward normal and 9 stickers. A sticker
// carries its centre and its 4 corners (a unit square in the face plane) in model
// space — corners are what pin orientation against the detected sticker shapes.
const MODEL = (() => {
  const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const dd = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
  const faces = [];
  for (let a = 0; a < 3; a++) {
    for (const s of [1, -1]) {
      const N = scale(ax[a], s), A = ax[(a + 1) % 3], B = ax[(a + 2) % 3];
      const faceCenter = scale(N, HALF);
      const stickers = [];
      for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
        const c = add(add(faceCenter, scale(A, i)), scale(B, j));
        const corners = dd.map(([di, dj]) => add(add(c, scale(A, di)), scale(B, dj)));
        stickers.push({ c, corners });
      }
      faces.push({ normal: N, stickers });
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

// Project the front-facing stickers (with centre + corners + their model 3D), for
// matching/refinement. front = outward normal toward the camera, all in view.
function projectModel(R, t, K) {
  const out = [];
  for (let fi = 0; fi < MODEL.length; fi++) {
    const f = MODEL[fi];
    const faceCenter = apply(R, t, scale(f.normal, HALF));
    if (dot(rot(R, f.normal), faceCenter) >= 0) continue;
    for (const st of f.stickers) {
      const C = apply(R, t, st.c);
      if (C[2] <= 0) continue;
      const k3 = st.corners, k2 = st.corners.map((p) => { const [x, y] = projPt(K, apply(R, t, p)); return { x, y }; });
      out.push({ fi, c3: st.c, c2: { x: projPt(K, C)[0], y: projPt(K, C)[1] }, k3, k2 });
    }
  }
  return out;
}

// Project front faces for drawing: [{ normal, cells:[{x,y}*9] }].
export function projectCube(R, t, K) {
  const out = [];
  for (const f of MODEL) {
    const faceCenter = apply(R, t, scale(f.normal, HALF));
    if (dot(rot(R, f.normal), faceCenter) >= 0) continue;
    const cells = f.stickers.map((st) => apply(R, t, st.c));
    if (cells.some((P) => P[2] <= 0)) continue;
    out.push({ normal: f.normal, cells: cells.map((P) => { const [x, y] = projPt(K, P); return { x, y }; }) });
  }
  return out;
}

// Front-face stickers' projected centre + corners, for tests / external callers.
export function projectStickers(R, t, K) {
  return projectModel(R, t, K).map((s) => ({ center: s.c2, corners: s.k2 }));
}

const cdist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Match a quad's centre to the nearest projected sticker (within tol). Returns the
// model sticker or null.
function matchSticker(q, model, tol) {
  let best = null, bd = tol;
  for (const m of model) { const d = cdist(q.center, m.c2); if (d < bd) { bd = d; best = m; } }
  return best;
}

// Mean distance from each of the quad's 4 corners to the nearest projected corner
// of its matched model sticker (shape agreement).
function cornerError(q, m) {
  let s = 0;
  for (const dc of q.corners) s += Math.min(...m.k2.map((p) => cdist(dc, p)));
  return s / q.corners.length;
}

// 3D<->2D correspondences from the four corners of every centre-matched sticker
// (centres alone underconstrain orientation). NOT corner-gated: we want a roughly-
// placed second face to get pulled in by the refine and lock, then SCORING judges
// the result. tol is loosened a touch so a rough seed can still grab the 2nd face.
function cornerCorrespondences(R, t, K, quads, tol) {
  const model = projectModel(R, t, K);
  const X = [], u = [];
  for (const q of quads) {
    const m = matchSticker(q, model, tol);
    if (!m) continue;
    for (const dc of q.corners) {
      let bi = 0, bd = Infinity;
      for (let k = 0; k < 4; k++) { const d = cdist(dc, m.k2[k]); if (d < bd) { bd = d; bi = k; } }
      X.push(m.k3[bi]); u.push([dc.x, dc.y]);
    }
  }
  return { X, u };
}

// Shape-aware, FACETED score. Count shape-agreeing matches per cube face; a corner-
// on pose is only credible if matches span >=2 faces (one face can't determine the
// 3D orientation, so a single-face extrapolation sends the other faces flying off —
// which must NOT score). Returns the total matched only when >=2 faces are covered.
function shapeScore(R, t, K, quads, tol, cornerTol) {
  const model = projectModel(R, t, K);
  const perFace = new Map();
  for (const q of quads) {
    const m = matchSticker(q, model, tol);
    if (m && cornerError(q, m) <= cornerTol) perFace.set(m.fi, (perFace.get(m.fi) || 0) + 1);
  }
  let total = 0, facesSeen = 0;
  for (const c of perFace.values()) { total += c; if (c >= 2) facesSeen++; }
  return facesSeen >= 2 ? total : 0;
}

// Estimate the cube pose: seed from each face fit, score each (shape-aware), refine
// the best over all matched sticker CORNERS with inlier re-selection. Returns
// { R, t, score, faces } or null.
export function estimateCubePose(fits, K, quads, opts = {}) {
  if (!fits.length) return null;
  const ref = fits.reduce((a, b) => (b.cells.length > a.cells.length ? b : a));
  const pitch = Math.hypot(ref.outline[1].x - ref.outline[0].x, ref.outline[1].y - ref.outline[0].y) / 3;
  const tol = (opts.matchFrac ?? 0.5) * pitch;
  const cornerTol = (opts.cornerFrac ?? 0.35) * pitch;

  let best = null;
  for (const fit of fits) {
    const { R, t } = seedFromFace(fit, K);
    const score = shapeScore(R, t, K, quads, tol, cornerTol);
    if (!best || score > best.score) best = { R, t, score };
  }

  // Refine, loosening the match radius on the first pass so a roughly-placed second
  // face gets grabbed and locked, then tightening.
  let { R, t } = best;
  for (let passNo = 0; passNo < 4; passNo++) {
    const { X, u } = cornerCorrespondences(R, t, K, quads, passNo === 0 ? tol * 1.8 : tol);
    if (X.length < 8) break;
    ({ R, t } = refinePnP(R, t, X, u, K));
  }
  const score = shapeScore(R, t, K, quads, tol, cornerTol);
  return { R, t, score, faces: projectCube(R, t, K) };
}

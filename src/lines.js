// Vanishing-point grouping for the line-based cube detector (step 1).
//
// A cube has edges along three mutually-orthogonal directions; under perspective
// projection each set of parallel 3D edges converges to one vanishing point (VP).
// So the Hough segments of a corner-on cube fall into (up to) three families, one
// per VP — and recovering those three VPs is what later yields the cube's rotation
// (cols of R = K⁻¹·vpᵢ). This file does ONLY the grouping: segments -> 3 families
// + their VPs, plus the per-line consistency error. Rotation/PnP come next.
//
// DOM-free and OpenCV-free (segments come in as plain {x1,y1,x2,y2}), so it's the
// offline-testable half — test/lines.test.mjs round-trips synthetic cube edges.
// The reads that produce the segments (Canny+Hough) are the browser-only layer.
//
// Algorithm: orientation-seeded k-means (the three face directions already cluster
// by 2D orientation — what the hue overlay shows) gives an initial 3-way split;
// then a few EM passes refine it on the true geometric criterion — each line is
// assigned to the VP it actually points at, and each VP is re-fit from its lines as
// the smallest eigenvector of Σ ℓℓᵀ. Lines that point at no VP are outliers
// (clutter: the hand, background, ceiling). A few random restarts guard the
// k-means against a bad local optimum; the partition explaining the most line
// length wins. All math runs in Hartley-normalized coordinates for conditioning.

import { jacobiEigenSymmetric } from './pose.js';

export const VP_DEFAULTS = {
  vpIters: 6,          // EM reassign/refit passes
  vpRestarts: 4,       // k-means seedings tried; keep the one explaining the most line length
  vpMaxErrorDeg: 3,    // a line farther than this (angle to its best VP) is an outlier
  vpMinLen: 14,        // segments shorter than this don't vote in the VP fit (still grouped/drawn)
};

const K_FAMILIES = 3;  // the three cube edge directions

// --- small helpers ---------------------------------------------------------

const DEG = Math.PI / 180;

// Homogeneous line (unit normal) through a segment's endpoints, plus its midpoint,
// length and unit direction. ℓ = p1 × p2, normalized so (a,b) is a unit normal.
function lineOf(seg) {
  const { x1, y1, x2, y2 } = seg;
  let a = y1 - y2, b = x2 - x1, c = x1 * y2 - x2 * y1;
  const nn = Math.hypot(a, b) || 1;
  a /= nn; b /= nn; c /= nn;
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  return { a, b, c, len, mid: [(x1 + x2) / 2, (y1 + y2) / 2], dir: [(x2 - x1) / len, (y2 - y1) / len] };
}

const mat3vec = (M, v) => [
  M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
  M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
  M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
];

// Deterministic PRNG (mulberry32) so restarts — and the tests — are reproducible.
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- public, reusable pieces (also unit-tested directly) -------------------

// Fit a vanishing point to a set of lines: the point v minimizing Σ wᵢ(ℓᵢ·v)² is
// the smallest eigenvector of M = Σ wᵢ ℓᵢℓᵢᵀ. Returns a homogeneous [vx,vy,vw]
// (unit norm); vw≈0 means a VP at infinity (parallel lines / near face-on view).
export function fitVanishingPoint(lines, weights) {
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let total = 0;
  for (let i = 0; i < lines.length; i++) {
    const w = weights ? weights[i] : 1;
    if (w <= 0) continue;
    const l = [lines[i].a, lines[i].b, lines[i].c];
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) M[r][s] += w * l[r] * l[s];
    total += w;
  }
  if (total === 0) return null;
  const { values, vectors } = jacobiEigenSymmetric(M);
  let mi = 0;
  for (let i = 1; i < 3; i++) if (values[i] < values[mi]) mi = i;
  return vectors[mi];
}

// Angular consistency of a line with a VP, in radians [0, π/2]: the angle between
// the segment's direction and the direction from its midpoint toward the VP. 0 = the
// line points exactly at the VP. Handles a VP at infinity (vw≈0 → direction (vx,vy)).
export function lineVPError(line, vp) {
  if (!vp) return Math.PI / 2;
  let dx, dy;
  if (Math.abs(vp[2]) < 1e-9) { dx = vp[0]; dy = vp[1]; }
  else { dx = vp[0] / vp[2] - line.mid[0]; dy = vp[1] / vp[2] - line.mid[1]; }
  const dn = Math.hypot(dx, dy) || 1;
  const cos = Math.min(1, Math.abs((line.dir[0] * dx + line.dir[1] * dy) / dn));
  return Math.acos(cos);
}

// --- grouping --------------------------------------------------------------

// Hartley-style similarity that centres endpoints on the origin at mean distance √2,
// for conditioning. Angles (hence VP errors) are similarity-invariant, so the whole
// pipeline runs normalized and only the output VPs are mapped back to pixels.
function normalizer(segments) {
  let mx = 0, my = 0, n = 0;
  for (const s of segments) { mx += s.x1 + s.x2; my += s.y1 + s.y2; n += 2; }
  mx /= n; my /= n;
  let d = 0;
  for (const s of segments) d += Math.hypot(s.x1 - mx, s.y1 - my) + Math.hypot(s.x2 - mx, s.y2 - my);
  d /= n;
  const sc = Math.SQRT2 / (d || 1);
  const T = (x, y) => ({ x: sc * (x - mx), y: sc * (y - my) });
  const Tinv = [[1 / sc, 0, mx], [0, 1 / sc, my], [0, 0, 1]]; // maps a homogeneous point back to pixels
  return { T, Tinv };
}

// Each undirected line orientation as a unit vector at DOUBLED angle, so θ and θ+π
// (the same line) coincide and circular k-means behaves on a full circle.
const orientPoint = (line) => {
  const phi = 2 * Math.atan2(line.dir[1], line.dir[0]);
  return [Math.cos(phi), Math.sin(phi)];
};

// k centres spread apart on the orientation circle (greedy farthest-point, weighted),
// or random for restarts. Returns initial centre unit-vectors.
function seedCentres(pts, w, k, rand) {
  if (rand) {
    const idx = [];
    while (idx.length < k) { const i = (rand() * pts.length) | 0; if (!idx.includes(i)) idx.push(i); }
    return idx.map((i) => pts[i].slice());
  }
  let start = 0;
  for (let i = 1; i < pts.length; i++) if (w[i] > w[start]) start = i;
  const centres = [pts[start].slice()];
  while (centres.length < k) {
    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < pts.length; i++) {
      let nearest = Infinity;
      for (const c of centres) nearest = Math.min(nearest, 1 - (pts[i][0] * c[0] + pts[i][1] * c[1]));
      const score = nearest * w[i];
      if (score > bestScore) { bestScore = score; best = i; }
    }
    centres.push(pts[best].slice());
  }
  return centres;
}

// Weighted circular k-means on the doubled-angle orientation points → an assignment.
function kmeansOrient(pts, w, k, centres0) {
  let centres = centres0.map((c) => c.slice());
  const assign = new Array(pts.length).fill(0);
  for (let it = 0; it < 10; it++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++) {
      let best = 0, bd = -Infinity;
      for (let c = 0; c < k; c++) {
        const dot = pts[i][0] * centres[c][0] + pts[i][1] * centres[c][1];
        if (dot > bd) { bd = dot; best = c; }
      }
      if (assign[i] !== best) { assign[i] = best; moved = true; }
    }
    const sx = new Array(k).fill(0), sy = new Array(k).fill(0);
    for (let i = 0; i < pts.length; i++) { sx[assign[i]] += w[i] * pts[i][0]; sy[assign[i]] += w[i] * pts[i][1]; }
    for (let c = 0; c < k; c++) {
      const n = Math.hypot(sx[c], sy[c]);
      if (n > 1e-9) centres[c] = [sx[c] / n, sy[c] / n];
    }
    if (!moved && it > 0) break;
  }
  return assign;
}

// EM: alternately fit a VP per family and reassign each line to its best VP (or to
// outlier if beyond maxErr). Returns { assign, vps } in the (normalized) frame.
function refineEM(lines, fitW, assign0, maxErr, iters) {
  let assign = assign0.slice();
  let vps = new Array(K_FAMILIES).fill(null);
  for (let it = 0; it < iters; it++) {
    for (let c = 0; c < K_FAMILIES; c++) {
      const sub = [], w = [];
      for (let i = 0; i < lines.length; i++) if (assign[i] === c) { sub.push(lines[i]); w.push(fitW[i]); }
      vps[c] = sub.length >= 2 ? fitVanishingPoint(sub, w) : null;
    }
    let moved = false;
    for (let i = 0; i < lines.length; i++) {
      let best = -1, bestE = Infinity;
      for (let c = 0; c < K_FAMILIES; c++) {
        if (!vps[c]) continue;
        const e = lineVPError(lines[i], vps[c]);
        if (e < bestE) { bestE = e; best = c; }
      }
      const a = bestE <= maxErr ? best : -1;
      if (a !== assign[i]) { assign[i] = a; moved = true; }
    }
    if (!moved && it > 0) break;
  }
  return { assign, vps };
}

// Total inlier line length explained by a partition — the score restarts compete on.
function inlierWeight(assign, lenW) {
  let s = 0;
  for (let i = 0; i < assign.length; i++) if (assign[i] >= 0) s += lenW[i];
  return s;
}

// Group Hough segments into the three cube-edge families and their vanishing points.
// Returns { families: [{ vp:[vx,vy,vw]|null, segments, meanErrorDeg }] (sorted by
// total length, desc), outliers: segments }. VPs are in pixel coordinates.
export function groupLineSegments(segments, opts = {}) {
  const o = { ...VP_DEFAULTS, ...opts };
  if (segments.length < K_FAMILIES * 2) return { families: [], outliers: segments.slice() };

  const { T, Tinv } = normalizer(segments);
  const lines = segments.map((s) => {
    const p1 = T(s.x1, s.y1), p2 = T(s.x2, s.y2);
    return lineOf({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y });
  });
  // Lengths are in the ORIGINAL frame (scale-invariant intent): vote/score by real px.
  const lenW = segments.map((s) => Math.hypot(s.x2 - s.x1, s.y2 - s.y1) || 1);
  const fitW = lenW.map((l) => (l >= o.vpMinLen ? l : 0));
  const maxErr = o.vpMaxErrorDeg * DEG;

  const orient = lines.map(orientPoint);
  let best = null, bestScore = -Infinity;
  for (let r = 0; r < Math.max(1, o.vpRestarts); r++) {
    const centres = seedCentres(orient, lenW, K_FAMILIES, r === 0 ? null : rng(r * 2654435761));
    const seed = kmeansOrient(orient, lenW, K_FAMILIES, centres);
    const em = refineEM(lines, fitW, seed, maxErr, o.vpIters);
    const score = inlierWeight(em.assign, lenW);
    if (score > bestScore) { bestScore = score; best = em; }
  }

  const families = [];
  for (let c = 0; c < K_FAMILIES; c++) {
    const segs = [], errs = [];
    for (let i = 0; i < lines.length; i++) {
      if (best.assign[i] !== c) continue;
      segs.push(segments[i]);
      errs.push(lineVPError(lines[i], best.vps[c]) / DEG);
    }
    const vp = best.vps[c] ? mat3vec(Tinv, best.vps[c]) : null;
    const meanErrorDeg = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : 0;
    families.push({ vp, segments: segs, totalLen: segs.reduce((a, s) => a + Math.hypot(s.x2 - s.x1, s.y2 - s.y1), 0), meanErrorDeg });
  }
  families.sort((a, b) => b.totalLen - a.totalLen);

  const outliers = [];
  for (let i = 0; i < segments.length; i++) if (best.assign[i] < 0) outliers.push(segments[i]);
  return { families, outliers };
}

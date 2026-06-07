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

const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
const normalize3 = (a) => { const n = norm3(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
// K helpers (K = {f,cx,cy}). KinvV: image point/VP -> camera-frame ray direction.
// KtL: image line -> its back-projected plane normal in camera frame (Kᵀℓ). Kd:
// a 3D direction -> its vanishing point in the image (K·d).
const KinvV = (K, v) => [(v[0] - K.cx * v[2]) / K.f, (v[1] - K.cy * v[2]) / K.f, v[2]];
const KtL = (K, l) => [K.f * l[0], K.f * l[1], K.cx * l[0] + K.cy * l[1] + l[2]];
const Kd = (K, d) => [K.f * d[0] + K.cx * d[2], K.f * d[1] + K.cy * d[2], d[2]];

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

// --- step 2: rotation from lines (orthogonal vanishing-point search) --------

export const ROT_DEFAULTS = {
  vpMaxErrorDeg: 3,    // a line within this angle of a VP is an inlier (shared with grouping)
  vpRansac: 600,       // RANSAC hypotheses for the orthogonal VP triple
  vpMinLen: 14,        // only segments at least this long seed/refine the model
};

// Fit a 3D direction common to a set of image lines: the direction d most orthogonal
// to all their back-projected plane normals n̂=normalize(Kᵀℓ) — the smallest
// eigenvector of Σ w n̂n̂ᵀ. Unit normals keep this well-conditioned (unlike fitting a
// VP in raw pixel coordinates), so it's what the refinement uses.
function fitDirection(normals, weights) {
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < normals.length; i++) {
    const w = weights[i], n = normals[i];
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) M[r][s] += w * n[r] * n[s];
  }
  const { values, vectors } = jacobiEigenSymmetric(M);
  let mi = 0;
  for (let i = 1; i < 3; i++) if (values[i] < values[mi]) mi = i;
  return vectors[mi];
}

// Assign each line to its nearest of three candidate DIRECTIONS (or outlier) and
// report total inlier length. Error is the 3D angle between the line's back-projected
// plane and the direction, asin|n̂·d̂| — a line truly along d lies in a plane
// containing d, so n̂⊥d. Measuring in 3D (not the 2D midpoint→VP angle of
// lineVPError) is distance-invariant: a far-away VP no longer loosely vacuums up
// lines that merely point roughly toward it.
function scoreTriple(normals, lens, dirs, maxErr) {
  let weight = 0;
  const assign = new Array(normals.length).fill(-1);
  const counts = [0, 0, 0];
  for (let i = 0; i < normals.length; i++) {
    let best = -1, bestE = Infinity;
    for (let c = 0; c < 3; c++) {
      const e = Math.asin(Math.min(1, Math.abs(dot3(normals[i], dirs[c]))));
      if (e < bestE) { bestE = e; best = c; }
    }
    if (bestE <= maxErr) { assign[i] = best; weight += lens[i]; counts[best]++; }
  }
  return { assign, weight, counts };
}

// A vanishing point at/near the principal point (direction ≈ optical axis) is a false
// attractor: every line through the image centre is roughly consistent with it, so it
// vacuums up support. A corner-on cube's three axes are all oblique (|d_z| ≈ 0.6), so
// reject any frame with an axis this close to the optical axis.
const MAX_AXIS_Z = 0.9;
const axisDegenerate = (dirs) => dirs.some((d) => Math.abs(d[2]) > MAX_AXIS_Z);

// Estimate the cube's camera rotation from Hough segments, given intrinsics K.
//
// A cube's three edge directions are mutually orthogonal, so their three vanishing
// points correspond (via K) to an orthonormal frame = the rotation R. We RANSAC for
// that frame: two lines give a first VP (direction d1); a third line, constrained to
// be orthogonal to d1, gives d2; d3 = d1×d2. The triple explaining the most line
// length wins, then it's refined on its inliers. Requiring orthogonality is the cube
// prior that rejects clutter — a convergent BACKGROUND bundle (books, a shelf) won't
// be orthogonal to the cube's other two directions, so it can't join the winning
// frame. R is only determined up to the cube's 24 symmetries, which is irrelevant for
// drawing an orientation (a cube looks the same under them) and is disambiguated later
// for the solve. Also returns the families (same shape as groupLineSegments) and a
// ROUGH pose (R + a translation guessed from the inliers' image centroid/spread) good
// enough to overlay an orientation wireframe; metric translation comes with the PnP.
export function estimateRotationFromLines(segments, K, opts = {}) {
  const o = { ...ROT_DEFAULTS, ...opts };
  if (segments.length < 6) return null;

  const lines = segments.map(lineOf);                       // pixel-coordinate lines
  const normals = lines.map((l) => normalize3(KtL(K, [l.a, l.b, l.c])));
  const lens = lines.map((l) => l.len);
  const maxErr = o.vpMaxErrorDeg * DEG;
  const longIdx = lines.map((_, i) => i).filter((i) => lines[i].len >= o.vpMinLen);
  if (longIdx.length < 4) return null;

  const rand = rng(0x9e3779b1);
  const pick = () => longIdx[(rand() * longIdx.length) | 0];

  let bestDirs = null, bestWeight = -Infinity;
  for (let it = 0; it < o.vpRansac; it++) {
    const i1 = pick(), i2 = pick();
    if (i1 === i2) continue;
    const v1 = cross3([lines[i1].a, lines[i1].b, lines[i1].c], [lines[i2].a, lines[i2].b, lines[i2].c]);
    if (norm3(v1) < 1e-9) continue;
    const d1 = normalize3(KinvV(K, v1));
    const i3 = pick();
    if (i3 === i1 || i3 === i2) continue;
    let d2 = cross3(d1, normals[i3]);          // ⊥ d1 and on line i3's plane
    if (norm3(d2) < 1e-6) continue;
    d2 = normalize3(d2);
    const d3 = cross3(d1, d2);
    const dirs0 = [d1, d2, d3];
    if (axisDegenerate(dirs0)) continue;
    const { weight, counts } = scoreTriple(normals, lens, dirs0, maxErr);
    // Require all three axes to have real support, so a single attractor can't win on
    // total weight alone — the cube shows all three edge directions corner-on.
    if (Math.min(counts[0], counts[1], counts[2]) < 2) continue;
    if (weight > bestWeight) { bestWeight = weight; bestDirs = dirs0; }
  }
  if (!bestDirs) return null;

  // Refine: reassign inliers, refit each direction (3D eigen-fit), re-orthonormalize.
  // The refit is UNGUARDED, so it can drift toward the optical-axis attractor; accept a
  // pass only if it keeps the frame non-degenerate with all three axes supported, else
  // keep the (balanced) RANSAC frame.
  let dirs = bestDirs;
  for (let pass = 0; pass < 2; pass++) {
    const { assign } = scoreTriple(normals, lens, dirs, maxErr);
    const refit = [];
    const support = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const ns = [], ws = [];
      for (let i = 0; i < lines.length; i++) if (assign[i] === c && lines[i].len >= o.vpMinLen) { ns.push(normals[i]); ws.push(lines[i].len); support[c] = (support[c] || 0) + lines[i].len; }
      refit[c] = ns.length >= 2 ? normalize3(fitDirection(ns, ws)) : dirs[c];
    }
    // Gram-Schmidt anchored on the best-supported axis, so noise doesn't tilt the frame.
    const order = [0, 1, 2].sort((a, b) => support[b] - support[a]);
    const a0 = normalize3(refit[order[0]]);
    let a1 = refit[order[1]];
    a1 = normalize3([a1[0] - dot3(a1, a0) * a0[0], a1[1] - dot3(a1, a0) * a0[1], a1[2] - dot3(a1, a0) * a0[2]]);
    const a2 = cross3(a0, a1);
    const nd = [];
    nd[order[0]] = a0; nd[order[1]] = a1; nd[order[2]] = a2;
    if (axisDegenerate(nd)) break;
    const chk = scoreTriple(normals, lens, nd, maxErr);
    if (Math.min(chk.counts[0], chk.counts[1], chk.counts[2]) < 2) break;
    dirs = nd;
  }
  // Enforce a proper rotation (det +1); flipping an axis is just another symmetry.
  if (dot3(dirs[0], cross3(dirs[1], dirs[2])) < 0) dirs[2] = [-dirs[2][0], -dirs[2][1], -dirs[2][2]];

  const vps = dirs.map((d) => Kd(K, d));
  const { assign } = scoreTriple(normals, lens, dirs, maxErr);
  const families = [0, 1, 2].map((c) => ({ vp: vps[c], segments: [] }));
  const outliers = [];
  for (let i = 0; i < segments.length; i++) {
    if (assign[i] >= 0) families[assign[i]].segments.push(segments[i]);
    else outliers.push(segments[i]);
  }
  const R = [
    [dirs[0][0], dirs[1][0], dirs[2][0]],
    [dirs[0][1], dirs[1][1], dirs[2][1]],
    [dirs[0][2], dirs[1][2], dirs[2][2]],
  ];

  return { R, pose: roughPose(R, families, K), families, outliers, inlierLen: bestWeight };
}

// A rough pose for the orientation overlay: keep R, and place a unit cube (edge 1) at
// the depth/position whose projection sits at the inlier segments' image centroid with
// a matching spread. Translation here is only a visualization aid — PnP gives the
// metric one — so the constants just need to land the wireframe near the cube.
function roughPose(R, families, K) {
  const segs = families.flatMap((f) => f.segments);
  if (segs.length < 2) return null;
  let mx = 0, my = 0;
  for (const s of segs) { mx += (s.x1 + s.x2) / 2; my += (s.y1 + s.y2) / 2; }
  mx /= segs.length; my /= segs.length;
  let rad = 0;
  for (const s of segs) rad += Math.hypot((s.x1 + s.x2) / 2 - mx, (s.y1 + s.y2) / 2 - my);
  rad = rad / segs.length || 1;
  const Z = K.f * 0.6 / rad;                 // edge-1 cube; ~match the inlier spread
  return { R, t: [(mx - K.cx) / K.f * Z, (my - K.cy) / K.f * Z, Z] };
}

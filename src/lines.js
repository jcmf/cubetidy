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

import { jacobiEigenSymmetric, refinePnP, project } from './pose.js';

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

// A rough pose to seed the metric refinement (and to draw the untrusted overlay):
// keep R, centre a unit cube on the inlier endpoints, and set depth from their image
// SPREAD (RMS distance from the centroid). Depth from segment LENGTH fails here —
// Hough breaks a close cube's grid into short fragments, so length underestimates the
// cube's size and puts it far too deep; the endpoint spread tracks the actual image
// footprint regardless of fragmentation. It's only a seed: the depth scan in
// recoverCubePose refines it, so the constant just needs to land within the scan band.
function roughPose(R, families, K) {
  const pts = families.flatMap((f) => f.segments).flatMap((s) => [[s.x1, s.y1], [s.x2, s.y2]]);
  if (pts.length < 4) return null;
  let mx = 0, my = 0;
  for (const [x, y] of pts) { mx += x; my += y; }
  mx /= pts.length; my /= pts.length;
  let rms = 0;
  for (const [x, y] of pts) rms += (x - mx) ** 2 + (y - my) ** 2;
  rms = Math.sqrt(rms / pts.length) || 1;
  const Z = K.f * 0.6 / rms;
  return { R, t: [(mx - K.cx) / K.f * Z, (my - K.cy) / K.f * Z, Z] };
}

// --- step 3: metric pose (lattice -> PnP) + confidence gate ----------------

export const POSE_DEFAULTS = {
  poseIters: 4,         // ICP-style associate→refine passes
  assocFrac: 0.6,       // associate a grid corner to lines within this fraction of a cell
  minCorr: 10,          // need at least this many grid-corner correspondences to lock
  maxReprojFrac: 0.1,   // ...reprojecting within this fraction of the cube edge (px)
  vpSweep: [3, 4, 5, 6],// inlier angles (deg) solveCubeFromLines tries; the best lock wins
};

// A 3×3 face is split by lines at these offsets (±½ boundaries, ±1/6 internal) along
// each in-plane axis; their pairwise crossings are the lattice corners.
const GRID_OFFSETS = [-0.5, -1 / 6, 1 / 6, 0.5];

// The three cube faces pointing toward the camera under rotation R: a face's outward
// normal s·e_k maps to camera-frame z = s·R[2][k]; it faces the camera when that is
// negative. Returns the three most camera-facing { k (axis), s (sign) }.
function visibleFaces(R) {
  const faces = [];
  for (let k = 0; k < 3; k++) for (const s of [-1, 1]) faces.push({ k, s, nz: s * R[2][k] });
  faces.sort((a, b) => a.nz - b.nz);
  return faces.slice(0, 3);
}

// Largest apparent cube-edge length in pixels under a pose — a foreshortening-robust
// scale for association thresholds and the reprojection gate.
function edgePixels(K, pose) {
  let m = 0;
  for (let k = 0; k < 3; k++) {
    const a = [0, 0, 0], b = [0, 0, 0];
    a[k] = 0.5; b[k] = -0.5;
    const pa = project(K, pose, a), pb = project(K, pose, b);
    m = Math.max(m, Math.hypot(pa[0] - pb[0], pa[1] - pb[1]));
  }
  return m;
}

// Nearest line to (x,y) among a family — but only lines whose SEGMENT (not just its
// infinite extension) reaches the point, within `margin`. A grid line spans the face,
// so a real corner falls on it; a clutter segment elsewhere is rejected even if its
// infinite line happens to pass nearby, which otherwise lets one stray segment shadow
// the correct grid line for many corners.
const nearestLine = (lines, x, y, margin) => {
  let best = null, bd = Infinity;
  for (const l of lines) {
    const s = (x - l.x1) * l.dir[0] + (y - l.y1) * l.dir[1]; // foot position along the segment
    if (s < -margin || s > l.len + margin) continue;
    const d = Math.abs(l.a * x + l.b * y + l.c);
    if (d < bd) { bd = d; best = l; }
  }
  return { line: best, dist: bd };
};
const intersect = (p, q) => {
  const w = p.a * q.b - q.a * p.b;
  if (Math.abs(w) < 1e-9) return null;
  return [(p.b * q.c - q.b * p.c) / w, (q.a * p.c - p.a * q.c) / w];
};

// Recover a metric 6-DoF cube pose from a step-2 rotation result, and decide whether
// to trust it (the confidence gate). With R fixed, the cube's grid model is projected
// at a rough pose; each projected lattice corner is associated to the nearest detected
// line of each of its two in-plane families and re-measured as their intersection —
// giving 3D↔2D correspondences fed to the existing refinePnP (a few associate→refine
// ICP passes). The cube center is invariant under the 24 cube symmetries, so the
// recovered translation is unambiguous even though R isn't. Returns { pose, count,
// reprojErr, edgePx, locked }; `locked` (enough corners reprojecting tightly) is the
// confidence gate — an ill-conditioned/wrong R yields few corners or a large residual
// and does not lock, so the overlay never commits to a confidently-wrong cube.
export function recoverCubePose(rot, K, opts = {}) {
  const o = { ...POSE_DEFAULTS, ...opts };
  if (!rot || !rot.pose) return null;

  const famLines = rot.families.map((f) => f.segments.map((s) => ({ ...lineOf(s), x1: s.x1, y1: s.y1 })));
  const faces = visibleFaces(rot.R);
  // Model lattice CORNERS (for the final PnP) and model grid LINES per direction (for
  // the depth search), both on the three visible faces.
  const models = [];
  const modelLines = [[], [], []];
  for (const f of faces) {
    const inplane = [0, 1, 2].filter((i) => i !== f.k);
    for (const ga of GRID_OFFSETS) for (const gb of GRID_OFFSETS) {
      const X = [0, 0, 0];
      X[f.k] = f.s * 0.5; X[inplane[0]] = ga; X[inplane[1]] = gb;
      models.push({ X, a: inplane[0], b: inplane[1] });
    }
    for (const along of inplane) {
      const off = inplane.find((i) => i !== along);
      for (const g of GRID_OFFSETS) {
        const A = [0, 0, 0], B = [0, 0, 0];
        A[f.k] = f.s * 0.5; B[f.k] = f.s * 0.5; A[off] = g; B[off] = g; A[along] = -0.5; B[along] = 0.5;
        modelLines[along].push([A, B]);
      }
    }
  }

  // Mean reprojection error of a correspondence set under a pose.
  const meanErr = (P, X3, q2) => {
    let s = 0;
    for (let i = 0; i < X3.length; i++) { const p = project(K, P, X3[i]); s += Math.hypot(p[0] - q2[i][0], p[1] - q2[i][1]); }
    return s / (X3.length || 1);
  };

  // Depth is the one DOF the rough pose gets badly wrong (foreshortening makes grid
  // lines look shorter, so length-based depth is biased) — and the periodic grid means
  // a bad scale lets ICP settle on a wrong alias. So before ICP, SCAN depth: slide the
  // cube along the centroid ray and pick the depth whose projected grid lines best
  // overlay the detected lines. A 1-D scan can't fall into a local minimum.
  const t0 = rot.pose.t;
  const ray = [t0[0] / t0[2], t0[1] / t0[2], 1];   // image-centroid ray; t = z·ray
  const projLine = (pose, A, B) => {
    const a = project(K, pose, A), b = project(K, pose, B);
    let na = a[1] - b[1], nb = b[0] - a[0], nc = a[0] * b[1] - b[0] * a[1];
    const n = Math.hypot(na, nb) || 1;
    return [na / n, nb / n, nc / n];
  };
  const lineMatchErr = (pose) => {
    let err = 0, w = 0;
    for (let c = 0; c < 3; c++) {
      const mls = modelLines[c].map(([A, B]) => projLine(pose, A, B));
      for (const s of rot.families[c].segments) {
        const mx = (s.x1 + s.x2) / 2, my = (s.y1 + s.y2) / 2, len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1) || 1;
        let d = Infinity;
        for (const ml of mls) d = Math.min(d, Math.abs(ml[0] * mx + ml[1] * my + ml[2]));
        err += Math.min(d, 40) * len; w += len;   // capped so a few stray lines don't dominate
      }
    }
    return err / (w || 1);
  };
  let bestZ = t0[2], bestZErr = Infinity;
  for (let i = 0; i <= 60; i++) {
    const z = t0[2] * (0.3 * Math.pow(3.5 / 0.3, i / 60));   // geometric scan 0.3×..3.5× rough depth
    const e = lineMatchErr({ R: rot.R, t: [z * ray[0], z * ray[1], z] });
    if (e < bestZErr) { bestZErr = e; bestZ = z; }
  }

  // ICP: associate, ROBUSTLY trim (the 3×3 grid is periodic and clutter can sneak into
  // a family, so a non-robust fit slides onto a wrong alias), refit, and stop if a pass
  // doesn't improve — keeping the best pose seen.
  let pose = { R: rot.R, t: [bestZ * ray[0], bestZ * ray[1], bestZ] };
  let best = { pose, X3: [], q2: [], err: Infinity };
  for (let iter = 0; iter < o.poseIters; iter++) {
    const thresh = Math.max(4, edgePixels(K, pose) / 3 * o.assocFrac);
    const X3 = [], q2 = [];
    for (const m of models) {
      const p = project(K, pose, m.X);
      const la = nearestLine(famLines[m.a], p[0], p[1], thresh);
      const lb = nearestLine(famLines[m.b], p[0], p[1], thresh);
      if (!la.line || !lb.line || la.dist > thresh || lb.dist > thresh) continue;
      const q = intersect(la.line, lb.line);
      if (!q || Math.hypot(q[0] - p[0], q[1] - p[1]) > thresh) continue;
      X3.push(m.X); q2.push(q);
    }
    if (X3.length < o.minCorr) break;
    // Trim correspondences far from the current pose (mis-associations) before refit.
    const r = X3.map((X, i) => { const p = project(K, pose, X); return Math.hypot(p[0] - q2[i][0], p[1] - q2[i][1]); });
    const med = [...r].sort((a, b) => a - b)[r.length >> 1] || 0;
    const cut = Math.max(3, 2.5 * med);
    const k = r.map((v, i) => (v <= cut ? i : -1)).filter((i) => i >= 0);
    const iX = k.map((i) => X3[i]), iq = k.map((i) => q2[i]);
    if (iX.length < o.minCorr) break;
    const next = refinePnP(pose.R, pose.t, iX, iq, K);
    if (!Number.isFinite(next.t[0]) || !Number.isFinite(next.t[2]) || next.t[2] <= 0) break;
    const err = meanErr(next, iX, iq);
    if (err < best.err) best = { pose: next, X3: iX, q2: iq, err };
    else break; // no improvement → converged or diverging; keep the best
    pose = next;
  }

  const edgePx = edgePixels(K, best.pose);
  const locked = best.X3.length >= o.minCorr && best.err <= o.maxReprojFrac * edgePx;
  return { pose: best.pose, count: best.X3.length, reprojErr: best.err, edgePx, locked };
}

// Full line→cube solve. The inlier angle that yields a good orthogonal frame varies
// frame to frame (noisier lines want a looser angle; a looser angle elsewhere lets a
// wrong frame win), and no single value is reliable across consecutive frames. So
// SWEEP a few angles, run rotation→pose for each, and keep the best result — a lock
// beats a non-lock, ties broken by lowest reprojection. A wrong frame fails to lock at
// every angle (too few corners), so the sweep can't manufacture a false lock. An
// explicit vpMaxErrorDeg (e.g. from the tuning panel) pins the sweep to that one value.
// Returns { rot, pose } (pose may be null/unlocked) or null if no rotation at all.
export function solveCubeFromLines(segments, K, opts = {}) {
  const sweep = opts.vpMaxErrorDeg != null ? [opts.vpMaxErrorDeg] : (opts.vpSweep ?? POSE_DEFAULTS.vpSweep);
  let best = null;
  for (const deg of sweep) {
    const o2 = { ...opts, vpMaxErrorDeg: deg };
    const rot = estimateRotationFromLines(segments, K, o2);
    if (!rot) continue;
    const pose = recoverCubePose(rot, K, o2);
    const cand = { rot, pose };
    if (!best) { best = cand; continue; }
    const la = best.pose && best.pose.locked, lb = pose && pose.locked;
    if (la !== lb) { if (lb) best = cand; }
    else if (la && lb) { if (pose.reprojErr < best.pose.reprojErr) best = cand; }
    else if ((pose ? pose.count : 0) > (best.pose ? best.pose.count : 0)) best = cand;
  }
  return best;
}

// --- temporal smoothing of the locked pose (24-fold-symmetry aware) ---------

// The per-frame lock is correct most frames but occasionally snaps to a wrong pose,
// so the overlay jitters between right and wrong. Smoothing fixes it — but a cube's
// rotation is only defined up to its 24 symmetries, so two *correct* consecutive
// frames can report numerically different R's; blending them naively would corrupt the
// pose. So: align each new R to the symmetry representative nearest the smoothed one,
// EMA-blend if it's consistent, and REJECT jumps (the wrong locks) — unless a new pose
// persists for a few frames, which means a real move and re-acquires.

export const POSE_SMOOTH_DEFAULTS = {
  poseAlpha: 0.4,        // EMA weight toward each accepted estimate (lower = steadier)
  poseMaxMiss: 8,        // hold the last pose through this many lock-less updates
  poseGateAngle: 22,     // a new pose >this° (after symmetry alignment) from the smoothed one is a jump
  poseGateTrans: 0.3,    // ...or whose centre moved >this × depth
  poseReacquire: 5,      // a jump sustained this many updates = a real move → re-acquire
};

// The 24 proper rotations of a cube: signed 3×3 permutation matrices with det +1. A
// geometric cube is identical under these, so a recovered R is only defined up to one.
export const CUBE_ROTATIONS = (() => {
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  const out = [];
  for (const p of perms) for (let s = 0; s < 8; s++) {
    const sg = [(s & 1) ? -1 : 1, (s & 2) ? -1 : 1, (s & 4) ? -1 : 1];
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) M[r][p[r]] = sg[r];
    const det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    if (det > 0) out.push(M);
  }
  return out;
})();

const matMul3 = (A, B) => {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
};
const frob = (A, B) => { let s = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += A[i][j] * B[i][j]; return s; };

// The cube-symmetry representative of R closest to ref (max trace(refᵀ·R·S) over the
// 24 symmetries S), so two representatives of the same orientation line up before EMA.
export function canonicalizeRotation(R, ref) {
  let best = R, bestScore = -Infinity;
  for (const S of CUBE_ROTATIONS) {
    const cand = matMul3(R, S);
    const score = frob(ref, cand);
    if (score > bestScore) { bestScore = score; best = cand; }
  }
  return best;
}

// Geodesic angle (degrees) between two rotations: acos((trace(AᵀB)−1)/2).
export function rotationAngleDeg(A, B) {
  return Math.acos(Math.max(-1, Math.min(1, (frob(A, B) - 1) / 2))) * 180 / Math.PI;
}

// Blend A toward B by alpha and re-orthonormalize (Gram-Schmidt) to a valid rotation.
// Only meaningful for nearby A,B (smoothing blends gated-consistent rotations).
export function blendRotation(A, B, alpha) {
  const M = A.map((row, i) => row.map((v, j) => v + alpha * (B[i][j] - v)));
  const col = (j) => [M[0][j], M[1][j], M[2][j]];
  const nrm = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const c0 = nrm(col(0));
  let c1 = col(1); c1 = nrm([c1[0] - dot(c1, c0) * c0[0], c1[1] - dot(c1, c0) * c0[1], c1[2] - dot(c1, c0) * c0[2]]);
  const c2 = cross(c0, c1);
  return [[c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], [c0[2], c1[2], c2[2]]];
}

// Is pose (R,t) consistent with reference (refR,refT)? Aligns R to the nearest
// symmetry rep of refR first, then checks the rotation angle and centre shift.
function poseConsistent(refR, refT, R, t, o) {
  const Rc = canonicalizeRotation(R, refR);
  const ang = rotationAngleDeg(refR, Rc);
  const dt = Math.hypot(t[0] - refT[0], t[1] - refT[1], t[2] - refT[2]) / (Math.abs(refT[2]) || 1);
  return { ok: ang <= o.poseGateAngle && dt <= o.poseGateTrans, Rc };
}

// One temporal-smoothing update. `prev` is the smoother state (or null), `P` the new
// locked pose {R,t} (or null when this frame didn't lock), `opts` overrides the
// defaults. Returns the new state { R, t, miss, reject, cand }. Pure → testable with a
// synthetic pose sequence.
export function smoothLinePose(prev, P, opts = {}) {
  const o = { ...POSE_SMOOTH_DEFAULTS, ...opts };
  if (!P) {  // no lock this update — hold the last pose through brief dropouts
    if (prev && prev.miss < o.poseMaxMiss) return { ...prev, miss: prev.miss + 1 };
    return null;
  }
  if (!prev) return { R: P.R, t: P.t.slice(), miss: 0, reject: 0, cand: null };

  const c = poseConsistent(prev.R, prev.t, P.R, P.t, o);
  if (c.ok) {
    return {
      R: blendRotation(prev.R, c.Rc, o.poseAlpha),
      t: prev.t.map((v, i) => v + o.poseAlpha * (P.t[i] - v)),
      miss: 0, reject: 0, cand: null,
    };
  }
  // A jump: keep holding the smoothed pose, but track whether the NEW pose persists
  // (consecutive mutually-consistent jumps = a real move) before re-acquiring; this is
  // what ignores scattered wrong locks while still following a genuine reorientation.
  const sustained = prev.cand && poseConsistent(prev.cand.R, prev.cand.t, P.R, P.t, o).ok;
  const cand = { R: P.R, t: P.t.slice(), n: sustained ? prev.cand.n + 1 : 1 };
  if (cand.n >= o.poseReacquire) return { R: P.R, t: P.t.slice(), miss: 0, reject: 0, cand: null };
  return { R: prev.R, t: prev.t, miss: 0, reject: (prev.reject || 0) + 1, cand };
}

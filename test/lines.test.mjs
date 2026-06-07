// Round-trips the vanishing-point grouping in src/lines.js against synthetic ground
// truth — the offline-checkable half of the line-based detector (step 1). We pick a
// camera pose, project the three families of cube edges (segments parallel to each
// cube axis) to the image, add clutter, then check groupLineSegments recovers three
// families whose vanishing points match the truth and that clutter is rejected.
//
// What this canNOT check: the Canny+Hough reads that produce real segments — that
// needs a camera and a physical cube, the usual split for this repo's pixel layer.
//
// Run: npm test

import { groupLineSegments, estimateRotationFromLines, recoverCubePose, fitVanishingPoint, lineVPError } from '../src/lines.js';
import { estimateIntrinsics, project } from '../src/pose.js';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`));
}

// Deterministic PRNG so the synthetic scene (and the test) is stable.
function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function rotationAxisAngle([ax, ay, az], angle) {
  const n = Math.hypot(ax, ay, az), x = ax / n, y = ay / n, z = az / n;
  const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
}

// True vanishing point of a cube-frame direction d: the image of its point at
// infinity, vp ∝ K·R·d (homogeneous; vw≈0 if that direction is parallel to the image
// plane). Returned unit-norm for sign-agnostic comparison.
function trueVP(K, R, d) {
  const dc = [
    R[0][0] * d[0] + R[0][1] * d[1] + R[0][2] * d[2],
    R[1][0] * d[0] + R[1][1] * d[1] + R[1][2] * d[2],
    R[2][0] * d[0] + R[2][1] * d[1] + R[2][2] * d[2],
  ];
  const v = [K.f * dc[0] + K.cx * dc[2], K.f * dc[1] + K.cy * dc[2], dc[2]];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

// Homogeneous points are equal up to scale AND sign: compare by |dot| of unit vectors.
function vpAgree(a, b) {
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const d = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return Math.abs(d); // 1 == identical direction
}

const W = 1280, H = 720;
const K = estimateIntrinsics(W, H, 60);
const AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// --- unit checks on the reusable pieces ------------------------------------

(() => {
  // Lines that genuinely pass through a known point recover it.
  const v = [400, 250, 1];
  const mk = (x, y) => { // segment from (x,y) toward the point
    const dx = v[0] - x, dy = v[1] - y, n = Math.hypot(dx, dy);
    return { x1: x, y1: y, x2: x + dx / n * 30, y2: y + dy / n * 30 };
  };
  const segs = [mk(100, 100), mk(700, 80), mk(200, 600), mk(900, 500)];
  const lines = segs.map((s) => {
    let a = s.y1 - s.y2, b = s.x2 - s.x1, c = s.x1 * s.y2 - s.x2 * s.y1;
    const nn = Math.hypot(a, b); a /= nn; b /= nn; c /= nn;
    const len = Math.hypot(s.x2 - s.x1, s.y2 - s.y1);
    return { a, b, c, len, mid: [(s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2], dir: [(s.x2 - s.x1) / len, (s.y2 - s.y1) / len] };
  });
  const vp = fitVanishingPoint(lines, lines.map((l) => l.len));
  check('fitVanishingPoint: recovers a finite concurrency point', vpAgree(vp, v) > 0.99999, `agree=${vpAgree(vp, v)}`);
  check('lineVPError: ~0 for a line pointing at the VP', lineVPError(lines[0], vp) < 1e-6);
  check('lineVPError: ~90° for a perpendicular line', Math.abs(lineVPError({ mid: [400, 250], dir: [0, 1] }, [400 + 100, 250, 1]) - Math.PI / 2) < 1e-6);
})();

(() => {
  // Parallel (horizontal) lines → VP at infinity along x.
  const lines = [10, 200, 500].map((k) => {
    const s = { x1: 0, y1: k, x2: 100, y2: k };
    let a = s.y1 - s.y2, b = s.x2 - s.x1, c = s.x1 * s.y2 - s.x2 * s.y1;
    const nn = Math.hypot(a, b); a /= nn; b /= nn; c /= nn;
    return { a, b, c, len: 100, mid: [50, k], dir: [1, 0] };
  });
  const vp = fitVanishingPoint(lines, lines.map((l) => l.len));
  check('fitVanishingPoint: parallel lines → VP at infinity', Math.abs(vp[2]) < 1e-6 && vpAgree(vp, [1, 0, 0]) > 0.9999, `vp=${vp}`);
})();

// --- full grouping on a synthetic corner-on cube ---------------------------

// Corner-on poses, where the three cube-edge VPs are well separated (~30°+ apart) —
// the regime the line detector targets. (Near-face-on views make two directions
// nearly parallel in the image, leaving their VPs a few degrees apart and genuinely
// hard to separate; recovering rotation from such a view is ill-posed anyway, so it
// is out of scope for step 1.)
for (const { tag, axis, angle, t } of [
  { tag: 'corner-on A', axis: [0.9, -1, 0.1], angle: 1.0, t: [0.4, -0.3, 9] },
  { tag: 'corner-on B', axis: [0.6, -1, 0.2], angle: 1.05, t: [-0.5, 0.4, 8] },
]) {
  const R = rotationAxisAngle(axis, angle);
  const truthVPs = AXES.map((d) => trueVP(K, R, d));

  // Precondition: confirm this scene actually has separated VPs, so the assertions
  // below are testing separability, not a coincidentally-easy or degenerate layout.
  let minSep = Infinity;
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) minSep = Math.min(minSep, Math.acos(vpAgree(truthVPs[i], truthVPs[j])) / Math.PI * 180);
  check(`${tag}: scene VPs are well separated (precondition)`, minSep > 18, `minSep=${minSep.toFixed(1)}°`);
  const rand = rng(7);
  const segments = [], label = [];

  // 12 segments per axis: short edges placed through the cube volume, projected.
  AXES.forEach((e, ai) => {
    for (let m = 0; m < 12; m++) {
      const base = [(rand() * 2 - 1) * 1.3, (rand() * 2 - 1) * 1.3, (rand() * 2 - 1) * 1.3];
      const h = 0.3 + rand() * 0.5;
      const p1 = project(K, { R, t }, [base[0] - e[0] * h, base[1] - e[1] * h, base[2] - e[2] * h]);
      const p2 = project(K, { R, t }, [base[0] + e[0] * h, base[1] + e[1] * h, base[2] + e[2] * h]);
      const jit = () => (rand() - 0.5) * 1.0; // ±0.5 px endpoint noise
      segments.push({ x1: p1[0] + jit(), y1: p1[1] + jit(), x2: p2[0] + jit(), y2: p2[1] + jit() });
      label.push(ai);
    }
  });
  // 10 clutter segments: random, not converging anywhere.
  for (let m = 0; m < 10; m++) {
    const x = rand() * W, y = rand() * H, a = rand() * Math.PI, len = 25 + rand() * 60;
    segments.push({ x1: x, y1: y, x2: x + Math.cos(a) * len, y2: y + Math.sin(a) * len });
    label.push(-1);
  }

  const g = groupLineSegments(segments, { vpMaxErrorDeg: 3 });

  check(`${tag}: returns three families`, g.families.length === 3, `got ${g.families.length}`);

  // Each true VP is matched by exactly one recovered family VP (distinct families).
  const matched = truthVPs.map((tv) => {
    let bi = -1, bg = 0;
    g.families.forEach((f, i) => { if (f.vp) { const a = vpAgree(f.vp, tv); if (a > bg) { bg = a; bi = i; } } });
    return { bi, bg };
  });
  const distinct = new Set(matched.map((m) => m.bi)).size === 3 && !matched.some((m) => m.bi < 0);
  check(`${tag}: each true VP matched by a distinct family`, distinct, JSON.stringify(matched.map((m) => m.bi)));
  check(`${tag}: matched VPs are accurate`, matched.every((m) => m.bg > 0.999), matched.map((m) => m.bg.toFixed(4)).join(','));

  // Grouping purity: for each true axis, ≥80% of its segments share one family.
  const family = new Array(segments.length).fill(-1);
  g.families.forEach((f, fi) => f.segments.forEach((s) => { family[segments.indexOf(s)] = fi; }));
  for (let ai = 0; ai < 3; ai++) {
    const idx = label.map((l, i) => (l === ai ? i : -1)).filter((i) => i >= 0);
    const counts = {};
    for (const i of idx) counts[family[i]] = (counts[family[i]] || 0) + 1;
    const top = Math.max(...Object.values(counts));
    check(`${tag}: axis ${ai} segments stay together (purity)`, top / idx.length >= 0.8, `${top}/${idx.length}`);
  }

  // Most clutter is rejected as outliers.
  const clutterIdx = label.map((l, i) => (l === -1 ? i : -1)).filter((i) => i >= 0);
  const rejected = clutterIdx.filter((i) => g.outliers.includes(segments[i])).length;
  check(`${tag}: most clutter rejected as outliers`, rejected >= clutterIdx.length * 0.6, `${rejected}/${clutterIdx.length}`);
}

// --- step 2: rotation from lines, with clutter that itself converges -----------

// How well Rest matches Rtrue up to the cube's 24 symmetries: match each estimated
// axis to the nearest true axis (by |dot|, directions are sign-free); require the
// three to map to distinct axes. Returns the worst alignment (1 = perfect), or 0 if
// two estimated axes collapse onto the same true axis.
function rotAgree(Rtrue, Rest) {
  const col = (M, j) => [M[0][j], M[1][j], M[2][j]];
  const used = new Set();
  let worst = 1;
  for (let j = 0; j < 3; j++) {
    const ce = col(Rest, j);
    let axis = -1, dot = 0;
    for (let k = 0; k < 3; k++) {
      const ct = col(Rtrue, k);
      const d = Math.abs(ce[0] * ct[0] + ce[1] * ct[1] + ce[2] * ct[2]);
      if (d > dot) { dot = d; axis = k; }
    }
    if (used.has(axis)) return 0;
    used.add(axis);
    worst = Math.min(worst, dot);
  }
  return worst;
}

(() => {
  const R = rotationAxisAngle([0.9, -1, 0.1], 1.0);
  const t = [0.4, -0.3, 9];
  const rand = rng(99);
  const segments = [], kind = []; // kind: 0..2 cube axis, 'bundle', 'rand'

  AXES.forEach((e, ai) => {
    for (let m = 0; m < 12; m++) {
      const b = [(rand() * 2 - 1) * 1.3, (rand() * 2 - 1) * 1.3, (rand() * 2 - 1) * 1.3];
      const h = 0.3 + rand() * 0.5;
      const p1 = project(K, { R, t }, [b[0] - e[0] * h, b[1] - e[1] * h, b[2] - e[2] * h]);
      const p2 = project(K, { R, t }, [b[0] + e[0] * h, b[1] + e[1] * h, b[2] + e[2] * h]);
      const j = () => (rand() - 0.5);
      segments.push({ x1: p1[0] + j(), y1: p1[1] + j(), x2: p2[0] + j(), y2: p2[1] + j() });
      kind.push(ai);
    }
  });

  // A convergent BACKGROUND bundle (like the books in corner1): 14 long parallel-ish
  // lines aimed at their own vanishing point, off in a corner — NOT orthogonal to the
  // cube frame, so the orthogonal search must reject it rather than adopt it as an axis.
  const bvp = [-500, 950]; // a clutter VP unrelated to the cube
  for (let m = 0; m < 14; m++) {
    const x = 60 + rand() * 240, y = 380 + rand() * 300;
    const dx = bvp[0] - x, dy = bvp[1] - y, n = Math.hypot(dx, dy), L = 60 + rand() * 50;
    segments.push({ x1: x, y1: y, x2: x + dx / n * L, y2: y + dy / n * L });
    kind.push('bundle');
  }
  // A little incoherent clutter too.
  for (let m = 0; m < 8; m++) {
    const x = rand() * W, y = rand() * H, a = rand() * Math.PI, L = 25 + rand() * 50;
    segments.push({ x1: x, y1: y, x2: x + Math.cos(a) * L, y2: y + Math.sin(a) * L });
    kind.push('rand');
  }

  const est = estimateRotationFromLines(segments, K, { vpMaxErrorDeg: 3 });
  check('rotation: estimate returned', !!est && !!est.R);
  check('rotation: matches truth up to cube symmetry', est && rotAgree(R, est.R) > 0.99, est ? `worst=${rotAgree(R, est.R).toFixed(4)}` : 'null');
  check('rotation: rough pose is in front of camera', est && est.pose && est.pose.t[2] > 0);

  // The convergent background bundle must NOT be adopted: most of its lines are outliers.
  if (est) {
    const inBundle = (s) => est.families.some((f) => f.segments.includes(s));
    let bundleInliers = 0, bundleTotal = 0;
    segments.forEach((s, i) => { if (kind[i] === 'bundle') { bundleTotal++; if (inBundle(s)) bundleInliers++; } });
    check('rotation: convergent clutter bundle rejected (orthogonality prior)', bundleInliers <= bundleTotal * 0.3, `${bundleInliers}/${bundleTotal} adopted`);

    // ...while the real cube edges are kept.
    let cubeInliers = 0, cubeTotal = 0;
    segments.forEach((s, i) => { if (typeof kind[i] === 'number') { cubeTotal++; if (inBundle(s)) cubeInliers++; } });
    check('rotation: cube edges kept as inliers', cubeInliers >= cubeTotal * 0.8, `${cubeInliers}/${cubeTotal}`);
  }
})();

// --- step 3: metric pose (lattice -> PnP) + confidence gate --------------------

// The three camera-facing cube faces under R (outward normal s·e_k has camera-frame
// z = s·R[2][k] < 0 when facing the camera) — mirrors lines.js visibleFaces.
function visFaces(R) {
  const f = [];
  for (let k = 0; k < 3; k++) for (const s of [-1, 1]) f.push({ k, s, nz: s * R[2][k] });
  return f.sort((a, b) => a.nz - b.nz).slice(0, 3);
}
const GRID = [-0.5, -1 / 6, 1 / 6, 0.5];

// Synthesize the projected grid-line segments of a corner-on cube: for each visible
// face, 4 lines along each in-plane axis (at the grid offsets), spanning the face.
function cubeGridSegments(R, t, rand) {
  const segs = [];
  for (const f of visFaces(R)) {
    const [a, b] = [0, 1, 2].filter((i) => i !== f.k);
    for (const [along, off] of [[a, b], [b, a]]) {
      for (const g of GRID) {
        const P1 = [0, 0, 0], P2 = [0, 0, 0];
        P1[f.k] = f.s * 0.5; P2[f.k] = f.s * 0.5;
        P1[off] = g; P2[off] = g;
        P1[along] = -0.5; P2[along] = 0.5;
        const u = project(K, { R, t }, P1), v = project(K, { R, t }, P2);
        const j = () => (rand() - 0.5);
        segs.push({ x1: u[0] + j(), y1: u[1] + j(), x2: v[0] + j(), y2: v[1] + j() });
      }
    }
  }
  return segs;
}

(() => {
  const R = rotationAxisAngle([0.9, -1, 0.1], 1.0);
  const t = [0.4, -0.3, 9];
  const rand = rng(2024);
  const segments = cubeGridSegments(R, t, rand);
  for (let m = 0; m < 10; m++) { // some incoherent clutter
    const x = rand() * W, y = rand() * H, ang = rand() * Math.PI, L = 25 + rand() * 50;
    segments.push({ x1: x, y1: y, x2: x + Math.cos(ang) * L, y2: y + Math.sin(ang) * L });
  }

  const rot = estimateRotationFromLines(segments, K, { vpMaxErrorDeg: 3 });
  const est = rot && recoverCubePose(rot, K, {});
  check('pose: locks onto the cube', !!est && est.locked, est ? `count=${est.count} err=${est.reprojErr.toFixed(2)} edge=${est.edgePx.toFixed(0)}` : 'null');
  if (est) {
    // The cube centre is invariant under the 24 symmetries, so t is unambiguous.
    const dt = Math.hypot(est.pose.t[0] - t[0], est.pose.t[1] - t[1], est.pose.t[2] - t[2]);
    check('pose: recovers metric translation', dt / Math.hypot(...t) < 0.05, `rel dt=${(dt / Math.hypot(...t)).toFixed(3)}`);
    check('pose: tight reprojection', est.reprojErr < 0.05 * est.edgePx, `err=${est.reprojErr.toFixed(2)} edge=${est.edgePx.toFixed(0)}`);
  }
})();

(() => {
  // Pure clutter: no cube → must NOT lock (the confidence gate earns its keep).
  const rand = rng(55), segments = [];
  for (let m = 0; m < 70; m++) {
    const x = rand() * W, y = rand() * H, ang = rand() * Math.PI, L = 25 + rand() * 70;
    segments.push({ x1: x, y1: y, x2: x + Math.cos(ang) * L, y2: y + Math.sin(ang) * L });
  }
  const rot = estimateRotationFromLines(segments, K, {});
  const est = rot && recoverCubePose(rot, K, {});
  check('pose: clutter-only scene does not lock', !est || !est.locked, est ? `count=${est.count} err=${est.reprojErr.toFixed(2)}` : 'no rot');
})();

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

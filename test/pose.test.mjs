// Round-trips the pose math in src/pose.js against synthetic ground truth — the
// offline-checkable half of pose estimation. We pick known camera poses, project a
// planar sticker grid through the pinhole model to get "detected" image points,
// then recover the homography and pose and assert they reproduce the truth.
//
// What this canNOT check: the pixel-side detection (OpenCV contour/quad finding)
// that produces real image points — that needs a camera and a physical cube.
//
// Run: npm test

import {
  estimateIntrinsics, intrinsicMatrix, intrinsicInverse,
  project, applyHomography, homographyDLT, poseFromHomography, reprojectionError,
} from '../src/pose.js';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`));
}
const approx = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

// Rotation about an arbitrary axis (Rodrigues), to build varied ground-truth poses.
function rotationAxisAngle([ax, ay, az], angle) {
  const n = Math.hypot(ax, ay, az);
  const x = ax / n, y = ay / n, z = az / n;
  const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
}
const isRotation = (R) => {
  // Columns orthonormal and right-handed (det = +1).
  const col = (j) => [R[0][j], R[1][j], R[2][j]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const c0 = col(0), c1 = col(1), c2 = col(2);
  const det =
    c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) -
    c0[1] * (c1[0] * c2[2] - c1[2] * c2[0]) +
    c0[2] * (c1[0] * c2[1] - c1[1] * c2[0]);
  return approx(dot(c0, c0), 1, 1e-6) && approx(dot(c1, c1), 1, 1e-6) &&
    approx(dot(c2, c2), 1, 1e-6) && approx(dot(c0, c1), 0, 1e-6) && approx(det, 1, 1e-6);
};

// A planar 3x3 sticker grid centred on the model origin (face-local units; one
// sticker ≈ 1 unit). These are the Z=0 model points a face-detector would supply.
const GRID = [];
for (let r = 0; r < 3; r++)
  for (let c = 0; c < 3; c++) GRID.push([c - 1, r - 1]);

const W = 1280, H = 720;
const K = estimateIntrinsics(W, H, 60);

// --- intrinsics & projection sanity ----------------------------------------

check('intrinsicMatrix · intrinsicInverse = I', (() => {
  const A = intrinsicMatrix(K), B = intrinsicInverse(K);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i][k] * B[k][j];
      if (!approx(s, i === j ? 1 : 0, 1e-9)) return false;
    }
  return true;
})());

check('a point on the optical axis projects to the principal point', (() => {
  const p = project(K, { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 0] }, [0, 0, 10]);
  return approx(p[0], K.cx, 1e-9) && approx(p[1], K.cy, 1e-9);
})());

// --- homography & pose round-trip across varied poses ----------------------

const POSES = [
  { axis: [0, 1, 0], angle: 0.0, t: [0, 0, 8] },     // fronto-parallel
  { axis: [0, 1, 0], angle: 0.5, t: [1, 0, 9] },     // yaw
  { axis: [1, 0, 0], angle: -0.4, t: [-1, 1, 10] },  // pitch
  { axis: [1, 1, 0], angle: 0.6, t: [2, -1, 11] },   // tilt + offset
  { axis: [0.3, 1, 0.2], angle: 0.9, t: [-2, 2, 12] }, // strong oblique
];

for (const { axis, angle, t } of POSES) {
  const tag = `pose axis=[${axis}] angle=${angle}`;
  const truth = { R: rotationAxisAngle(axis, angle), t };
  // "Detected" image points: project each grid point (Z=0) through the truth pose.
  const corr = GRID.map((X) => ({ X, u: project(K, truth, [X[0], X[1], 0]) }));

  const Hh = homographyDLT(corr);
  const hErr = Math.max(...corr.map(({ X, u }) => {
    const [x, y] = applyHomography(Hh, X);
    return Math.hypot(x - u[0], y - u[1]);
  }));
  check(`${tag}: homography maps model->image`, hErr < 1e-4, `maxErr=${hErr}`);

  const est = poseFromHomography(Hh, K);
  check(`${tag}: recovered R is a proper rotation`, isRotation(est.R));

  const reErr = reprojectionError(K, est, corr);
  check(`${tag}: pose reprojects model points`, reErr < 1e-4, `meanErr=${reErr}`);

  // Pose is unique here (planar, in front of camera), so it should match truth.
  const tErr = Math.hypot(est.t[0] - t[0], est.t[1] - t[1], est.t[2] - t[2]);
  check(`${tag}: recovered translation matches truth`, tErr < 1e-3, `dt=${tErr}`);
  let rErr = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    rErr = Math.max(rErr, Math.abs(est.R[i][j] - truth.R[i][j]));
  check(`${tag}: recovered rotation matches truth`, rErr < 1e-3, `dR=${rErr}`);
}

// --- robustness: recovery survives small detection noise -------------------

(() => {
  const truth = { R: rotationAxisAngle([0.2, 1, 0.1], 0.7), t: [1, -1, 10] };
  const corr = GRID.map((X) => {
    const u = project(K, truth, [X[0], X[1], 0]);
    // ±0.5 px jitter, deterministic so the test is stable.
    const j = (n) => ((Math.sin(n * 12.9898) * 43758.5453) % 1) - 0.5;
    return { X, u: [u[0] + j(X[0] + 1) , u[1] + j(X[1] + 2)] };
  });
  const est = poseFromHomography(homographyDLT(corr), K);
  const reErr = reprojectionError(K, est, corr);
  check('noisy detection: pose still reprojects within ~1px', reErr < 1.5, `meanErr=${reErr}`);
})();

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

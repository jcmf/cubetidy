// Pose math for the auto-scan: recover a cube's orientation from detected image
// features, and project 3D model points back onto the frame for AR overlays.
//
// This is the OFFLINE-TESTABLE half of pose estimation (test/pose.test.mjs round-
// trips synthetic poses). The pixel-side detection that produces the 2D points —
// OpenCV contour/quad finding — lives elsewhere and can only be checked in a real
// browser, the same split the rest of this repo uses for its sampling layer.
//
// Camera model is a standard pinhole, OpenCV-compatible so cv.solvePnP results
// drop in interchangeably: image = K · (R·X + t), then divide by z. K is a simple
// intrinsic { f, cx, cy } with square pixels and the principal point at the image
// centre. We can't read the true focal length from getUserMedia, so we ESTIMATE it
// from an assumed horizontal field of view (~60°); that is plenty for overlay
// alignment and can be refined later (e.g. by minimising reprojection error).

// --- small linear-algebra helpers ------------------------------------------

const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const normalize3 = (a) => scale3(a, 1 / (norm3(a) || 1));

function mat3mul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}

function mat3vec(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

function mat3inv(M) {
  const [a, b, c] = M[0], [d, e, f] = M[1], [g, h, i] = M[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const id = 1 / det;
  return [
    [A * id, (c * h - b * i) * id, (b * f - c * e) * id],
    [B * id, (a * i - c * g) * id, (c * d - a * f) * id],
    [C * id, (b * g - a * h) * id, (a * e - b * d) * id],
  ];
}

// Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.
// Returns eigenvalues and eigenvectors (as rows). Used to find the null vector of
// AᵀA in the homography DLT; n is small (9) so the O(n³)/sweep cost is irrelevant.
// Exported so the vanishing-point fit in lines.js can reuse it (smallest eigenvector
// of Σ ℓℓᵀ for a family of lines).
export function jacobiEigenSymmetric(input, maxSweeps = 100) {
  const n = input.length;
  const A = input.map((row) => row.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  const offDiagNorm = () => {
    let s = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) s += A[p][q] * A[p][q];
    return s;
  };

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (offDiagNorm() < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        // Givens rotation angle that zeroes A[p][q] (Numerical Recipes form).
        const theta = (A[q][q] - A[p][p]) / (2 * apq);
        const sign = theta >= 0 ? 1 : -1;
        const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // A ← Jᵀ A J : update columns p,q then rows p,q.
        for (let i = 0; i < n; i++) {
          const aip = A[i][p], aiq = A[i][q];
          A[i][p] = c * aip - s * aiq;
          A[i][q] = s * aip + c * aiq;
        }
        for (let i = 0; i < n; i++) {
          const api = A[p][i], aqi = A[q][i];
          A[p][i] = c * api - s * aqi;
          A[q][i] = s * api + c * aqi;
        }
        // V ← V J (accumulate eigenvectors as columns).
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] = c * vip - s * viq;
          V[i][q] = s * vip + c * viq;
        }
      }
    }
  }

  const values = A.map((row, i) => row[i]);
  const vectors = values.map((_, col) => V.map((row) => row[col]));
  return { values, vectors };
}

// --- public API ------------------------------------------------------------

// Approximate pinhole intrinsics for a frame of the given size. `fovDeg` is the
// assumed horizontal field of view; the focal length follows from it.
export function estimateIntrinsics(width, height, fovDeg = 60) {
  const f = (width / 2) / Math.tan((fovDeg * Math.PI) / 180 / 2);
  return { f, cx: width / 2, cy: height / 2 };
}

// 3x3 intrinsic matrix and its inverse, for code that needs the matrix form.
export function intrinsicMatrix({ f, cx, cy }) {
  return [[f, 0, cx], [0, f, cy], [0, 0, 1]];
}
export function intrinsicInverse({ f, cx, cy }) {
  return [[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]];
}

// Project a 3D world point through pose { R, t } and intrinsics K to pixels.
export function project(K, pose, X) {
  const Xc = [
    pose.R[0][0] * X[0] + pose.R[0][1] * X[1] + pose.R[0][2] * X[2] + pose.t[0],
    pose.R[1][0] * X[0] + pose.R[1][1] * X[1] + pose.R[1][2] * X[2] + pose.t[1],
    pose.R[2][0] * X[0] + pose.R[2][1] * X[1] + pose.R[2][2] * X[2] + pose.t[2],
  ];
  return [K.f * Xc[0] / Xc[2] + K.cx, K.f * Xc[1] / Xc[2] + K.cy];
}

// Map a planar point [x, y] through a 3x3 homography to [u, v].
export function applyHomography(H, [x, y]) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [
    (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
    (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
  ];
}

// Hartley normalization: similarity that centres `points` on the origin with mean
// distance √2. Returns the 3x3 transform T (point_normalized = T · point_h).
function normalizing2D(points) {
  let cx = 0, cy = 0;
  for (const [x, y] of points) { cx += x; cy += y; }
  cx /= points.length; cy /= points.length;
  let meanDist = 0;
  for (const [x, y] of points) meanDist += Math.hypot(x - cx, y - cy);
  meanDist /= points.length;
  const s = Math.SQRT2 / (meanDist || 1);
  return [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]];
}

// Direct Linear Transform homography from >=4 planar correspondences, each
// { X: [x, y] (model plane), u: [u, v] (image) }. Uses normalized DLT for
// conditioning. Returns the 3x3 H mapping model -> image, scaled so H[2][2] = 1.
export function homographyDLT(correspondences) {
  const src = correspondences.map((c) => c.X);
  const dst = correspondences.map((c) => c.u);
  const Ts = normalizing2D(src);
  const Td = normalizing2D(dst);
  const sn = src.map((p) => applyHomography(Ts, p));
  const dn = dst.map((p) => applyHomography(Td, p));

  const A = [];
  for (let i = 0; i < sn.length; i++) {
    const [x, y] = sn[i];
    const [u, v] = dn[i];
    A.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    A.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  // Smallest eigenvector of AᵀA is the least-squares null vector.
  const AtA = Array.from({ length: 9 }, (_, r) =>
    Array.from({ length: 9 }, (_, c) => {
      let s = 0;
      for (let k = 0; k < A.length; k++) s += A[k][r] * A[k][c];
      return s;
    }));
  const { values, vectors } = jacobiEigenSymmetric(AtA);
  let min = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[min]) min = i;
  const h = vectors[min];
  const Hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];

  // Denormalize: H = Td⁻¹ · Hn · Ts.
  const H = mat3mul(mat3inv(Td), mat3mul(Hn, Ts));
  const scale = H[2][2];
  return H.map((row) => row.map((v) => v / scale));
}

// Recover camera pose { R, t } of a planar patch from its model->image homography
// and intrinsics K. The model plane is taken as Z = 0; t is in the same units as
// the model coordinates. R is orthonormalized to the nearest rotation, and the
// sign chosen so the patch sits in front of the camera (t_z > 0).
export function poseFromHomography(H, K) {
  const Kinv = intrinsicInverse(K);
  const M = mat3mul(Kinv, H);
  let h1 = [M[0][0], M[1][0], M[2][0]];
  let h2 = [M[0][1], M[1][1], M[2][1]];
  let h3 = [M[0][2], M[1][2], M[2][2]];

  // Scale so the rotation columns are unit length (average the two estimates).
  const lambda = 2 / (norm3(h1) + norm3(h2));
  let r1 = scale3(h1, lambda);
  let r2 = scale3(h2, lambda);
  let t = scale3(h3, lambda);
  if (t[2] < 0) { r1 = scale3(r1, -1); r2 = scale3(r2, -1); t = scale3(t, -1); }

  // Orthonormalize (Gram-Schmidt) to the nearest valid rotation.
  const c1 = normalize3(r1);
  const c2 = normalize3(sub3(r2, scale3(c1, dot3(r2, c1))));
  const c3 = cross3(c1, c2);
  const R = [
    [c1[0], c2[0], c3[0]],
    [c1[1], c2[1], c3[1]],
    [c1[2], c2[2], c3[2]],
  ];
  return { R, t };
}

// --- iterative pose refinement (PnP via Gauss-Newton) ----------------------

const skew = (a) => [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]];

// Exponential map so(3) -> SO(3) (Rodrigues), for an incremental rotation.
function expSO3(w) {
  const th = Math.hypot(w[0], w[1], w[2]);
  if (th < 1e-12) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const k = [w[0] / th, w[1] / th, w[2] / th];
  const Kx = skew(k), K2 = mat3mul(Kx, Kx), s = Math.sin(th), c = 1 - Math.cos(th);
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  return I.map((row, i) => row.map((e, j) => e + s * Kx[i][j] + c * K2[i][j]));
}

// Solve a small dense linear system A x = b by Gauss-Jordan; null if singular.
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
    }
  }
  return b.map((_, i) => M[i][n] / M[i][i]);
}

// Refine a camera pose { R, t } to minimize reprojection error of 3D->2D
// correspondences (Gauss-Newton over a 6-vector [rotation, translation]). Far more
// stable than a single-face seed because it fuses all visible stickers at once.
export function refinePnP(R0, t0, pts3D, pts2D, K, iters = 12) {
  let R = R0.map((r) => r.slice()), t = [...t0];
  for (let it = 0; it < iters; it++) {
    const H = Array.from({ length: 6 }, () => new Array(6).fill(0));
    const g = new Array(6).fill(0);
    for (let i = 0; i < pts3D.length; i++) {
      const Rx = mat3vec(R, pts3D[i]);            // R * X (rotation acts before t)
      const Xc = [Rx[0] + t[0], Rx[1] + t[1], Rx[2] + t[2]];
      if (Xc[2] <= 1e-6) continue;
      const invz = 1 / Xc[2];
      const ru = K.f * Xc[0] * invz + K.cx - pts2D[i][0];
      const rv = K.f * Xc[1] * invz + K.cy - pts2D[i][1];
      const dpu = [K.f * invz, 0, -K.f * Xc[0] * invz * invz];
      const dpv = [0, K.f * invz, -K.f * Xc[1] * invz * invz];
      // dXc/d[w,t] = [ -skew(R*X) | I ]
      const a = Rx;
      const dXc = [
        [0, a[2], -a[1], 1, 0, 0],
        [-a[2], 0, a[0], 0, 1, 0],
        [a[1], -a[0], 0, 0, 0, 1],
      ];
      const Ju = new Array(6), Jv = new Array(6);
      for (let k = 0; k < 6; k++) {
        Ju[k] = dpu[0] * dXc[0][k] + dpu[1] * dXc[1][k] + dpu[2] * dXc[2][k];
        Jv[k] = dpv[0] * dXc[0][k] + dpv[1] * dXc[1][k] + dpv[2] * dXc[2][k];
      }
      for (let r = 0; r < 6; r++) {
        g[r] += Ju[r] * ru + Jv[r] * rv;
        for (let c = 0; c < 6; c++) H[r][c] += Ju[r] * Ju[c] + Jv[r] * Jv[c];
      }
    }
    const delta = solveLinear(H, g.map((x) => -x));
    if (!delta) break;
    R = mat3mul(expSO3([delta[0], delta[1], delta[2]]), R);
    t = [t[0] + delta[3], t[1] + delta[4], t[2] + delta[5]];
    if (Math.hypot(...delta) < 1e-10) break;
  }
  return { R, t };
}

// Mean reprojection error (pixels) of model<->image correspondences under a pose.
// A convenient gauge-free check: the recovered pose is good iff this is small.
export function reprojectionError(K, pose, correspondences) {
  let sum = 0;
  for (const { X, u } of correspondences) {
    const [px, py] = project(K, pose, [X[0], X[1], X[2] ?? 0]);
    sum += Math.hypot(px - u[0], py - u[1]);
  }
  return sum / correspondences.length;
}

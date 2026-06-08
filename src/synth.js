// Synthetic cube scenes — build a cube at a KNOWN 6-DoF pose and draw it onto any
// 2D canvas context. The browser-safe, dependency-light core shared by the node CLI
// (tools/synth-cube.mjs, which adds skia-canvas + file I/O) and the in-page ?synth
// harness (main.js). Reuses the detector's own project()/grid model (pose.js +
// GRID_OFFSETS, kept in sync with src/lines.js) so a rendered frame and its recorded
// ground truth can't drift apart.
//
// buildCubeScene(opts) is PURE (no canvas) — it returns the known properties (pose,
// K, per-sticker colours, projected corners/cells). drawScene(ctx, scene, opts)
// rasterizes it: a black cube body with inset colour stickers, the gaps forming the
// grid edges the line detector reads, plus optional blur/noise to mimic a capture.

import { estimateIntrinsics, project } from './pose.js';
import { FACE_DISPLAY } from './colors.js';

// Sticker boundaries along each in-plane axis — MUST match GRID_OFFSETS in
// src/lines.js (the 3×3 lattice the detector reprojects against). Four boundaries =
// three stickers of pitch 1/3 across the unit face [-0.5, 0.5].
export const GRID_OFFSETS = [-0.5, -1 / 6, 1 / 6, 0.5];

// Which face letter sits on each signed axis of the cube's own frame. Arbitrary but
// fixed (and recorded in the ground truth); colours come from FACE_DISPLAY so a
// rendered face matches the app's notion of that colour.
const FACE_LETTER = { '0+': 'R', '0-': 'L', '1+': 'U', '1-': 'D', '2+': 'F', '2-': 'B' };
const LETTERS = ['U', 'D', 'F', 'B', 'R', 'L'];

// --- small helpers ----------------------------------------------------------

function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Rodrigues: rotation matrix for a right-handed turn of `angle` rad about `axis`.
export function rotationAxisAngle([ax, ay, az], angle) {
  const n = Math.hypot(ax, ay, az) || 1;
  const x = ax / n, y = ay / n, z = az / n;
  const c = Math.cos(angle), s = Math.sin(angle), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
}

// The cube faces actually facing the camera at this pose — the ones drawScene should
// paint. A planar face is visible when its outward normal points back toward the camera:
// Nc·Cc < 0, where Nc = R·(s·e_k) is the normal and Cc = R·(s/2·e_k)+t the face centre,
// both in camera coords. This is the PERSPECTIVE-correct test and DELIBERATELY differs
// from the detector's orthographic visibleFaces in src/lines.js (s·R[2][k]<0, always 3):
// that optical-axis approximation misjudges a near-edge-on face when the cube is close
// AND laterally offset — a back face whose centre is off-axis can still score
// s·R[2][k]<0, so the old code drew a PHANTOM face over the real ones (e.g. ?synth dist=3
// with a tx/ty offset painted the down/up face across the top). The detector keeps the
// simple always-corner-on prior (its scan is corner-on); the renderer must be physically
// correct. Returns the genuine visible set (1–3 faces), farthest-first for painter order.
function visibleFaces(R, t) {
  const faces = [];
  for (let k = 0; k < 3; k++) for (const s of [-1, 1]) {
    const nx = R[0][k] * s, ny = R[1][k] * s, nz = R[2][k] * s;             // normal, camera coords
    const cx = nx * 0.5 + t[0], cy = ny * 0.5 + t[1], cz = nz * 0.5 + t[2]; // face centre, camera coords
    if (nx * cx + ny * cy + nz * cz < 0) faces.push({ k, s, depth: cz });
  }
  faces.sort((a, b) => b.depth - a.depth); // farthest first
  return faces;
}

// Largest apparent cube-edge length in px under a pose (matches edgePixels in lines.js).
function edgePixels(K, pose) {
  let m = 0;
  for (let k = 0; k < 3; k++) {
    const a = [0, 0, 0], b = [0, 0, 0]; a[k] = 0.5; b[k] = -0.5;
    const pa = project(K, pose, a), pb = project(K, pose, b);
    m = Math.max(m, Math.hypot(pa[0] - pb[0], pa[1] - pb[1]));
  }
  return m;
}

const hexToRgb = (h) => {
  const n = parseInt(h.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

// Resolve the rotation axis from any of: an explicit array, a "x,y,z" string, or
// separate axisX/axisY/axisZ knobs (how the ?synth panel feeds it). Default is a
// corner-on view with three faces visible and well-separated vanishing points.
function resolveAxis(opts) {
  if (Array.isArray(opts.axis)) return opts.axis;
  if (opts.axis != null) return String(opts.axis).split(',').map(Number);
  if (opts.axisX != null || opts.axisY != null || opts.axisZ != null)
    return [+opts.axisX || 0, +opts.axisY || 0, +opts.axisZ || 0];
  return [0.9, -1, 0.1];
}

// --- scene (the known properties) -------------------------------------------

// Build a synthetic cube scene at a known pose. Pure — no canvas — so it can be
// imported by tests/verifiers. Returns everything needed to render AND everything
// needed to grade a detector: K, the ground-truth pose, the visible faces with
// per-sticker colours, and the projected lattice / cell centres / cube corners.
export function buildCubeScene(opts = {}) {
  const width = +opts.width || 1280;
  const height = +opts.height || 720;
  const fovDeg = +opts.fovDeg || 60;
  const K = estimateIntrinsics(width, height, fovDeg);

  // Pose: explicit R wins; otherwise axis + angle (a degenerate zero axis falls back
  // to the default so a panel slider parked at 0,0,0 still renders a cube).
  const axis = resolveAxis(opts);
  const axisOk = Math.hypot(axis[0], axis[1], axis[2]) > 1e-6;
  const useAxis = axisOk ? axis : [0.9, -1, 0.1];
  const angle = opts.angleDeg != null ? (+opts.angleDeg * Math.PI / 180) : 1.0;
  const R = opts.R || rotationAxisAngle(useAxis, angle);
  const dist = +opts.dist || 6;
  const t = opts.t || [+opts.tx || 0, +opts.ty || 0, dist];
  const pose = { R, t };

  const faces = visibleFaces(R, t);
  const seed = +opts.scramble || 0;
  const rand = rng(seed || 1);

  // Per-visible-face 3×3 sticker grid (row-major in the two in-plane axes). Solved =
  // every sticker the face's own colour; scramble>0 = random colour per sticker
  // (visually rich for the detector; the exact colours are recorded as truth).
  const stickers = [];      // flat list for rendering: { quad3D[4], rgb, hex, letter }
  const facelets = [];      // per-face structured truth
  for (const f of faces) {
    const letter = FACE_LETTER[`${f.k}${f.s > 0 ? '+' : '-'}`];
    const inplane = [0, 1, 2].filter((i) => i !== f.k);
    const grid = { k: f.k, s: f.s, faceLetter: letter, cells: [] };
    for (let r = 0; r < 3; r++) {
      const row = [];
      for (let c = 0; c < 3; c++) {
        const cellLetter = seed ? LETTERS[Math.floor(rand() * 6)] : letter;
        const hex = FACE_DISPLAY[cellLetter];
        const rgb = hexToRgb(hex);
        // Sticker spans [GRID_OFFSETS[c], GRID_OFFSETS[c+1]] along inplane[0] and
        // [GRID_OFFSETS[r], GRID_OFFSETS[r+1]] along inplane[1]. Inset for the gap is
        // applied at draw time (the scene keeps the full cell so cell centres are
        // exact). Corners ordered CCW.
        const a0 = GRID_OFFSETS[c], a1 = GRID_OFFSETS[c + 1];
        const b0 = GRID_OFFSETS[r], b1 = GRID_OFFSETS[r + 1];
        const corner = (a, b) => { const X = [0, 0, 0]; X[f.k] = f.s * 0.5; X[inplane[0]] = a; X[inplane[1]] = b; return X; };
        const quad3D = [corner(a0, b0), corner(a1, b0), corner(a1, b1), corner(a0, b1)];
        const centre3D = corner((a0 + a1) / 2, (b0 + b1) / 2);
        const quad2D = quad3D.map((X) => project(K, pose, X));
        const centre2D = project(K, pose, centre3D);
        stickers.push({ quad3D, quad2D, rgb, hex, letter: cellLetter, k: f.k, s: f.s });
        row.push({ letter: cellLetter, hex, rgb, centre2D, axes: inplane });
      }
      grid.cells.push(row);
    }
    facelets.push(grid);
  }

  // Cube corners and the full visible grid lattice, for grading reprojection.
  const cubeCorners3D = [];
  for (const sx of [-0.5, 0.5]) for (const sy of [-0.5, 0.5]) for (const sz of [-0.5, 0.5]) cubeCorners3D.push([sx, sy, sz]);
  const cubeCorners2D = cubeCorners3D.map((X) => project(K, pose, X));

  return {
    width, height, fovDeg, K,
    pose,
    truth: {
      axis: opts.R ? null : useAxis, angleDeg: opts.R ? null : angle * 180 / Math.PI,
      R, t, dist,
      visibleFaces: faces.map((f) => ({ k: f.k, s: f.s, letter: FACE_LETTER[`${f.k}${f.s > 0 ? '+' : '-'}`] })),
      edgePx: edgePixels(K, pose),
      cubeCorners2D, facelets,
    },
    stickers,
  };
}

// --- raster -----------------------------------------------------------------

// Inset a projected quad toward its centroid by fraction `g` of the cell — the black
// gap between stickers. Working in image space is a fine approximation at these
// foreshortenings and keeps the gap visually uniform.
function insetQuad(quad2D, g) {
  const cx = (quad2D[0][0] + quad2D[1][0] + quad2D[2][0] + quad2D[3][0]) / 4;
  const cy = (quad2D[0][1] + quad2D[1][1] + quad2D[2][1] + quad2D[3][1]) / 4;
  return quad2D.map(([x, y]) => [x + (cx - x) * g, y + (cy - y) * g]);
}

// Draw a scene onto a 2D context (browser Canvas or skia-canvas — same API). Black
// cube body shows through the inter-sticker gaps as grid edges; `bg` fills behind it;
// optional blur/noise approximate a capture. Fills the whole context, so the caller
// just supplies a correctly-sized canvas.
//
// DETERMINISM: we rasterize ourselves (no-AA scanline fill at SxS supersampling, then
// box-average down) and putImageData ONCE, rather than using ctx.fill(). Canvas2D PATH
// ANTIALIASING is not reproducible — each backend (browser GPU, browser CPU under
// willReadFrequently, skia/node) shades the ~1px edge ramp differently, so Canny/Hough
// sees different segments and the recovered pose drifts (a frame that locks offline can
// alias live). Our fill + integer downsample are pure IEEE-754 arithmetic, so the SAME
// url params yield byte-identical pixels — hence identical segments and pose — in the
// live canvas, the offscreen Save canvas, and the node tools. Supersampling (vs a bare
// no-AA fill) keeps SMOOTH edges so the detector behaves as it did under native AA;
// plain staircased edges fragment diagonal grid lines and made some poses alias.
// (Cross-engine caveat: Math.sin/cos can differ by an ULP off V8, so a non-Chrome
// browser may shift an occasional edge pixel — still vastly more stable than native AA,
// and node↔Chrome are both V8 = exact.)
export function drawScene(ctx, scene, opts = {}) {
  const { width, height, stickers } = scene;
  const gap = opts.gap != null ? +opts.gap : 0.1;
  const [br, bgg, bb] = parseColor(opts.bg || '#15151a');
  const S = Math.max(1, Math.min(4, Math.round(+opts.ss || 3))); // supersample factor (AA quality)
  const SW = width * S, SH = height * S;

  // High-res no-AA buffer (reused across frames to avoid per-frame multi-MB allocation).
  const big = ssBuffer(SW * SH * 4);
  for (let i = 0, n = SW * SH * 4; i < n; i += 4) { big[i] = br; big[i + 1] = bgg; big[i + 2] = bb; big[i + 3] = 255; }
  const scaled = (pts) => pts.map(([x, y]) => [x * S, y * S]);
  // Cube body: a black silhouette per visible face (so there's a black rim and black
  // gaps for the grid edges). Painter order doesn't matter — a convex cube's visible
  // faces don't overlap in projection.
  for (const face of groupByFace(stickers)) fillPolygon(big, SW, SH, scaled(face.outline), 10, 10, 10);
  for (const s of stickers) fillPolygon(big, SW, SH, scaled(insetQuad(s.quad2D, gap)), s.rgb[0], s.rgb[1], s.rgb[2]);

  // Downsample SxS -> 1x by averaging (the deterministic antialiasing).
  const img = ctx.createImageData(width, height);
  const data = img.data, inv = 1 / (S * S);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let r = 0, g = 0, b = 0;
    for (let dy = 0; dy < S; dy++) { let o = ((y * S + dy) * SW + x * S) * 4; for (let dx = 0; dx < S; dx++, o += 4) { r += big[o]; g += big[o + 1]; b += big[o + 2]; } }
    const o = (y * width + x) * 4; data[o] = r * inv; data[o + 1] = g * inv; data[o + 2] = b * inv; data[o + 3] = 255;
  }

  // `imgBlur` (alias `blur` for the CLI) — named so it doesn't collide with the
  // detector's own Canny `blur` knob when ?synth&detect runs both tuning panels. A
  // deterministic separable box blur (3 passes ≈ Gaussian) — NOT ctx.filter, which is
  // backend-dependent and would reintroduce the non-determinism this function avoids.
  const blurPx = +(opts.imgBlur ?? opts.blur ?? 0);
  if (blurPx > 0) boxBlur(data, width, height, Math.max(1, Math.round(blurPx)));
  if (opts.noise && +opts.noise > 0) addNoise(data, +opts.noise, opts.seed);

  ctx.putImageData(img, 0, 0);
}

// Reused supersample scratch buffer — drawScene runs every rAF frame in ?synth, so we
// must not allocate a multi-MB typed array each call. Grows monotonically.
let _ssBuf = null;
function ssBuffer(n) { if (!_ssBuf || _ssBuf.length < n) _ssBuf = new Uint8ClampedArray(n); return _ssBuf; }

// Parse the handful of bg forms we accept (named black/white, #rgb, #rrggbb, rgb(...))
// to [r,g,b]; unknown falls back to the default dark grey. Kept tiny on purpose — the
// software fill needs numeric colours, and synth only ever uses these.
function parseColor(c) {
  if (c === 'black') return [0, 0, 0];
  if (c === 'white') return [255, 255, 255];
  let m = /^#([0-9a-f]{3})$/i.exec(c);
  if (m) return [...m[1]].map((h) => parseInt(h + h, 16));
  m = /^#([0-9a-f]{6})$/i.exec(c);
  if (m) return [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16));
  m = /^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i.exec(c);
  if (m) return [+m[1], +m[2], +m[3]];
  return [21, 21, 26];
}

// No-AA scanline polygon fill into an RGBA buffer. Pixel centres at (x+0.5, y+0.5);
// each scanline collects edge crossings, sorts them, and fills between pairs (even-odd
// rule, fine for the convex quads/hulls synth draws). Half-open at the upper vertex
// (yi<=yc<yj) so shared edges don't double-fill. Pure arithmetic ⇒ backend-independent.
function fillPolygon(data, W, H, pts, r, g, b) {
  const n = pts.length;
  let minY = Infinity, maxY = -Infinity;
  for (const p of pts) { if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1]; }
  const y0 = Math.max(0, Math.ceil(minY - 0.5)), y1 = Math.min(H - 1, Math.floor(maxY - 0.5));
  const xs = [];
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    xs.length = 0;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const yi = pts[i][1], yj = pts[j][1];
      if ((yi <= yc && yj > yc) || (yj <= yc && yi > yc)) {
        const t = (yc - yi) / (yj - yi);
        xs.push(pts[i][0] + t * (pts[j][0] - pts[i][0]));
      }
    }
    xs.sort((a, c) => a - c);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xL = Math.max(0, Math.ceil(xs[k] - 0.5)), xR = Math.min(W - 1, Math.floor(xs[k + 1] - 0.5));
      for (let x = xL; x <= xR; x++) { const o = (y * W + x) * 4; data[o] = r; data[o + 1] = g; data[o + 2] = b; }
    }
  }
}

// Deterministic separable box blur, `passes` (3 ≈ Gaussian) of radius `r`, in place.
function boxBlur(data, W, H, r, passes = 3) {
  const tmp = new Float64Array(W * H * 3);
  const win = 2 * r + 1;
  for (let p = 0; p < passes; p++) {
    // horizontal
    for (let y = 0; y < H; y++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let x = -r; x <= r; x++) sum += data[(y * W + Math.min(W - 1, Math.max(0, x))) * 4 + c];
        for (let x = 0; x < W; x++) {
          tmp[(y * W + x) * 3 + c] = sum / win;
          const xo = Math.max(0, x - r), xi = Math.min(W - 1, x + r + 1);
          sum += data[(y * W + xi) * 4 + c] - data[(y * W + xo) * 4 + c];
        }
      }
    }
    // vertical
    for (let x = 0; x < W; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let y = -r; y <= r; y++) sum += tmp[(Math.min(H - 1, Math.max(0, y)) * W + x) * 3 + c];
        for (let y = 0; y < H; y++) {
          data[(y * W + x) * 4 + c] = sum / win;
          const yo = Math.max(0, y - r), yi = Math.min(H - 1, y + r + 1);
          sum += tmp[(yi * W + x) * 3 + c] - tmp[(yo * W + x) * 3 + c];
        }
      }
    }
  }
}

// The black-body silhouette of each visible face = the convex hull of its projected
// sticker corners (the face is convex), padded outward a touch for a clean rim.
function groupByFace(stickers) {
  const byFace = new Map();
  for (const s of stickers) {
    const key = `${s.k}${s.s}`;
    if (!byFace.has(key)) byFace.set(key, []);
    byFace.get(key).push(s);
  }
  return [...byFace.values()].map((cells) => ({ outline: padHull(convexHull(cells.flatMap((c) => c.quad2D)), 2) }));
}

function convexHull(points) {
  const pts = points.map(([x, y]) => [x, y]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function padHull(hull, px) {
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
  return hull.map(([x, y]) => { const d = Math.hypot(x - cx, y - cy) || 1; return [x + (x - cx) / d * px, y + (y - cy) / d * px]; });
}

// Add seeded triangular (~Gaussian) luminance noise in place, on the RGBA buffer. Seeded
// so it's deterministic: the same `seed` (default 7) reproduces the same field.
function addNoise(data, sigma, seed) {
  const rand = rng((+seed || 7) ^ 0x9e3779b9);
  for (let i = 0; i < data.length; i += 4) {
    const n = (rand() + rand() - 1) * sigma * 2; // triangular ≈ gaussian, scaled to ~sigma
    data[i] = Math.max(0, Math.min(255, data[i] + n));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + n));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + n));
  }
}

// Generate SYNTHETIC cube frames with KNOWN ground truth — a controlled cube at a
// chosen 6-DoF pose, rendered to a PNG, with a sidecar JSON recording exactly what
// was drawn (pose, intrinsics, per-sticker colours, projected corners/cells).
//
// Why this exists: the line-detector splits into an offline-testable GEOMETRY half
// (groupLineSegments → solveCubeFromLines → recoverCubePose, round-tripped in
// test/lines.test.mjs from pre-projected segments) and a PIXEL half (Canny + Hough
// in src/detect.js) that until now could only be run on the real captured frames in
// samples/ — which have NO ground truth, so you can eyeball the wireframe but can't
// measure pose error. A synthetic frame closes that: it has a known pose, so the
// REAL pixel pipeline can be pushed end-to-end and the recovered pose checked
// against truth (tools/synth-smoke.mjs does exactly that).
//
// The renderer reuses the detector's own project()/grid model (src/pose.js +
// GRID_OFFSETS below, kept in sync with src/lines.js), so the image and its ground
// truth can't drift apart. Output is intentionally clean — a black cube body with
// inset colour stickers, the gaps forming the grid edges Hough reads — plus optional
// blur/noise to approximate a real capture.
//
//   node tools/synth-cube.mjs [out.png] [key=val ...]
//   node tools/synth-cube.mjs samples/synth/a.png angleDeg=57 dist=5 axis=0.9,-1,0.1
//
// Knobs: width,height,fovDeg | axis=x,y,z angleDeg dist tx ty (pose) |
//        scramble=<seed> (random per-sticker colours; 0=solved) |
//        gap bg blur noise (rendering). Writes <out>.png and <out>.truth.json.

import { Canvas } from 'skia-canvas';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { estimateIntrinsics, project } from '../src/pose.js';
import { FACE_DISPLAY } from '../src/colors.js';

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

// The three cube faces pointing toward the camera under R (replicates the private
// visibleFaces in src/lines.js): a face normal s·e_k maps to camera z = s·R[2][k];
// it faces the camera when that is negative.
function visibleFaces(R) {
  const faces = [];
  for (let k = 0; k < 3; k++) for (const s of [-1, 1]) faces.push({ k, s, nz: s * R[2][k] });
  faces.sort((a, b) => a.nz - b.nz);
  return faces.slice(0, 3);
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

  // Pose: explicit R wins; otherwise axis + angle. Default is a corner-on view with
  // three faces visible and well-separated vanishing points (the detector's regime).
  const axis = opts.axis
    ? String(opts.axis).split(',').map(Number)
    : [0.9, -1, 0.1];
  const angle = opts.angleDeg != null ? (+opts.angleDeg * Math.PI / 180) : 1.0;
  const R = opts.R || rotationAxisAngle(axis, angle);
  const dist = +opts.dist || 6;
  const t = opts.t || [+opts.tx || 0, +opts.ty || 0, dist];
  const pose = { R, t };

  const faces = visibleFaces(R);
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
        // applied at render time (the scene keeps the full cell so cell centres are
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
      axis: opts.R ? null : axis, angleDeg: opts.R ? null : angle * 180 / Math.PI,
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

// Render a scene to a skia Canvas. Black cube body shows through the inter-sticker
// gaps as grid edges; `bg` fills behind it; optional blur/noise approximate a capture.
export function renderScene(scene, opts = {}) {
  const { width, height, stickers } = scene;
  const gap = opts.gap != null ? +opts.gap : 0.1;
  const bg = opts.bg || '#15151a';
  const canvas = new Canvas(width, height);
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = bg === 'black' ? '#000' : bg === 'white' ? '#fff' : bg;
  ctx.fillRect(0, 0, width, height);

  // Cube body: fill each visible face solid black slightly OUTSIDE the stickers, so
  // there is a black rim and black gaps for the grid edges. Painter order doesn't
  // matter (a convex cube's visible faces don't overlap in projection).
  const fillPoly = (pts, style) => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fillStyle = style;
    ctx.fill();
  };
  // Body silhouette per face from its outer cell corners (cells 0 and 8 share the face
  // corners). Simpler: draw a black quad over the face extent, then the stickers.
  for (const st of groupByFace(stickers)) fillPoly(st.outline, '#0a0a0a');
  for (const s of stickers) {
    const q = insetQuad(s.quad2D, gap);
    fillPoly(q, `rgb(${s.rgb[0]},${s.rgb[1]},${s.rgb[2]})`);
  }

  if (opts.blur && +opts.blur > 0) { ctx.filter = `blur(${+opts.blur}px)`; ctx.drawImage(canvas, 0, 0); ctx.filter = 'none'; }
  if (opts.noise && +opts.noise > 0) addNoise(ctx, width, height, +opts.noise, opts.seed);
  return canvas;
}

// The four outer corners of each visible face (for the black body quad).
function groupByFace(stickers) {
  const byFace = new Map();
  for (const s of stickers) {
    const key = `${s.k}${s.s}`;
    if (!byFace.has(key)) byFace.set(key, []);
    byFace.get(key).push(s);
  }
  return [...byFace.values()].map((cells) => {
    // Outline = convex hull of all this face's projected sticker corners. The face is
    // convex so the hull is its silhouette; pad outward a touch for a clean rim.
    const pts = cells.flatMap((c) => c.quad2D);
    return { outline: padHull(convexHull(pts), 2) };
  });
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

function addNoise(ctx, w, h, sigma, seed) {
  const img = ctx.getImageData(0, 0, w, h);
  const rand = rng((+seed || 7) ^ 0x9e3779b9);
  for (let i = 0; i < img.data.length; i += 4) {
    // Box-Muller-ish: sum of two uniforms ≈ triangular, scaled to ~sigma.
    const n = (rand() + rand() - 1) * sigma * 2;
    img.data[i] = Math.max(0, Math.min(255, img.data[i] + n));
    img.data[i + 1] = Math.max(0, Math.min(255, img.data[i + 1] + n));
    img.data[i + 2] = Math.max(0, Math.min(255, img.data[i + 2] + n));
  }
  ctx.putImageData(img, 0, 0);
}

// --- CLI --------------------------------------------------------------------

function parseArgs(argv) {
  const opts = {}, paths = [];
  for (const a of argv) {
    const m = a.match(/^([A-Za-z]\w*)=(.+)$/);
    if (m) opts[m[1]] = /^-?\d*\.?\d+$/.test(m[2]) ? parseFloat(m[2]) : m[2];
    else paths.push(a);
  }
  return { opts, paths };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const { opts, paths } = parseArgs(process.argv.slice(2));
  const out = paths[0] || 'samples/synth/cube.png';
  const scene = buildCubeScene(opts);
  const canvas = renderScene(scene, opts);

  mkdirSync(dirname(out), { recursive: true });
  await canvas.toFile(out);
  const truthPath = out.replace(/\.(png|jpe?g|webp)$/i, '') + '.truth.json';
  writeFileSync(truthPath, JSON.stringify(scene.truth, null, 2));

  const tr = scene.truth;
  console.log(`${basename(out)} ${scene.width}x${scene.height} | faces ${tr.visibleFaces.map((f) => f.letter).join('')} | `
    + `angle ${tr.angleDeg == null ? 'R' : tr.angleDeg.toFixed(0) + '°'} dist ${tr.dist} | edge ${tr.edgePx.toFixed(0)}px -> ${out} (+ ${basename(truthPath)})`);
}

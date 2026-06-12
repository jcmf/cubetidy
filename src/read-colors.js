// Read sticker colours off a locked cube pose — the bridge from the line detector's
// 6-DoF pose to the colour classifier (colors.js). Given a frame and a pose {R, t},
// project each genuinely-visible face's nine cell centres, sample a small patch of
// REAL 3D points inside each cell (perspective-correct, so oblique faces sample the
// right pixels), and reduce each patch to a robust median RGB plus the confidence
// signals a temporal accumulator needs: how frontal the face is, how much of the
// patch landed in frame, and how scattered the patch was (edges / glare / noise).
//
// Pure and browser-safe: ImageData is duck-typed ({ width, height, data }), no DOM,
// no OpenCV — so the SAME code runs in the live harness (main.js), the offline bench
// (tools/synth-bench.mjs), and plain `npm test` (test/read-colors.test.mjs renders a
// synthetic frame through a stub context — the one place the pixel-sampling layer
// gets node coverage, because sampling at the TRUTH pose needs no detector).
//
// COORDINATES / the 24-fold ambiguity: faces are identified by their body-frame axis
// (k ∈ 0..2, s = ±1) and cells are row-major over the face's two in-plane axes —
// exactly buildCubeScene's truth facelets, so reading under a truth-aligned
// (canonicalizeRotation'd) pose grades index-for-index. Which face LETTER a (k, s)
// is can't be known from geometry (R is only defined up to the cube's 24 symmetries);
// it stays stable across frames while the smoothed pose holds one symmetry rep, and
// is resolved for real once the six CENTRE colours are classified.

import { project, visibleCubeFaces } from './pose.js';
import { rgbToLab, labDistance, FACE_DISPLAY } from './colors.js';

export const READ_DEFAULTS = {
  readInner: 0.5,    // sampled patch spans this fraction of the cell about its centre
                     //   (0.5 stays inside the sticker at the default 0.1 gap with room
                     //   for a couple of pixels of pose error / blur bleed)
  readGrid: 5,       // patch = readGrid × readGrid sample points
  readMinFront: 0.15,// a face more edge-on than this view-cosine reads garbage → weight 0
  readSdScale: 30,   // patch spread (8-bit RGB sd) at which a cell's weight is halved —
                     //   a clean sticker interior measures ≪ this; an edge or glare
                     //   boundary through the patch measures ≫
};

// Cell centres along a face's two in-plane axes: midpoints of the GRID_OFFSETS
// sticker boundaries ([-1/2, -1/6, 1/6, 1/2] in synth.js / lines.js).
const CELL_CENTRES = [-1 / 3, 0, 1 / 3];

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const stddev = (a) => {
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
};

// Read the sticker colours visible under pose {R, t}. Returns { faces: [{ k, s,
// cells }] } with 9 cells row-major per face, each { rgb: {r,g,b}|null, px: [x,y],
// sd, inFrame, front, weight }. `weight` folds the confidence signals into one
// number (0 = unusable) so accumulation and grading share a single quality notion:
//   front (view cosine, gated)  ×  inFrame (patch fraction on-screen)
//   ÷ (1 + (sd / readSdScale)²) (patch scatter: edges, glare boundaries, noise).
export function readStickerColors(imageData, K, pose, opts = {}) {
  const o = { ...READ_DEFAULTS, ...opts };
  const { width, height, data } = imageData;
  const n = Math.max(2, o.readGrid | 0);
  const half = o.readInner * (1 / 6); // cell half-width is 1/6 of the unit cube
  const { R, t } = pose;

  const faces = [];
  for (const f of visibleCubeFaces(R, t)) {
    const inplane = [0, 1, 2].filter((i) => i !== f.k);
    const nx = R[0][f.k] * f.s, ny = R[1][f.k] * f.s, nz = R[2][f.k] * f.s; // face normal, camera coords
    const cells = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const at = (du, dv) => {
        const X = [0, 0, 0];
        X[f.k] = f.s * 0.5;
        X[inplane[0]] = CELL_CENTRES[c] + du;
        X[inplane[1]] = CELL_CENTRES[r] + dv;
        return X;
      };
      const C = at(0, 0);
      const Xc = [
        R[0][0] * C[0] + R[0][1] * C[1] + R[0][2] * C[2] + t[0],
        R[1][0] * C[0] + R[1][1] * C[1] + R[1][2] * C[2] + t[1],
        R[2][0] * C[0] + R[2][1] * C[1] + R[2][2] * C[2] + t[2],
      ];
      const dn = Math.hypot(Xc[0], Xc[1], Xc[2]) || 1;
      // View cosine at the cell: 1 = face-on, 0 = edge-on (and <0 shouldn't survive
      // the visibility test, but a cell behind the camera is dropped outright).
      const front = -(nx * Xc[0] + ny * Xc[1] + nz * Xc[2]) / dn;
      const px = project(K, pose, C);
      if (Xc[2] <= 1e-6) { cells.push({ rgb: null, px, sd: 0, inFrame: 0, front, weight: 0 }); continue; }

      const rs = [], gs = [], bs = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const q = project(K, pose, at((i / (n - 1) - 0.5) * 2 * half, (j / (n - 1) - 0.5) * 2 * half));
        const x = Math.round(q[0]), y = Math.round(q[1]);
        if (x < 0 || y < 0 || x >= width || y >= height) continue;
        const off = (y * width + x) * 4;
        rs.push(data[off]); gs.push(data[off + 1]); bs.push(data[off + 2]);
      }
      const inFrame = rs.length / (n * n);
      const rgb = rs.length ? { r: median(rs), g: median(gs), b: median(bs) } : null;
      const sd = rs.length > 1 ? (stddev(rs) + stddev(gs) + stddev(bs)) / 3 : 0;
      const weight = rgb && front >= o.readMinFront
        ? (front * inFrame) / (1 + (sd / o.readSdScale) ** 2)
        : 0;
      cells.push({ rgb, px, sd, inFrame, front, weight });
    }
    faces.push({ k: f.k, s: f.s, cells });
  }
  return { faces };
}

// Nearest standard face colour (FACE_DISPLAY, compared in Lab) — for grading reads
// against synth ground truth and for debug overlays. NOT the production classifier:
// real frames go through the balanced k-means in colors.js once all 54 stickers are
// gathered, precisely because fixed references break under real lighting.
const PALETTE = Object.entries(FACE_DISPLAY).map(([letter, hex]) => {
  const v = parseInt(hex.slice(1), 16);
  return { letter, lab: rgbToLab({ r: (v >> 16) & 255, g: (v >> 8) & 255, b: v & 255 }) };
});
export function nearestFaceLetter(rgb) {
  const lab = rgbToLab(rgb);
  let best = null, bd = Infinity;
  for (const p of PALETTE) {
    const d = labDistance(lab, p.lab);
    if (d < bd) { bd = d; best = p.letter; }
  }
  return best;
}

// --- temporal accumulation ---------------------------------------------------
//
// Glare and shading are VIEW-dependent, so the strongest lever against them is
// averaging each sticker across frames as the hand drifts — the tracked-pose
// generalization of aggregateFaces' multi-pass mean. Cells are keyed by body-frame
// face (k, s) + index, which is only stable while the lock is continuous (the
// smoothed pose holds one symmetry rep): the CALLER must restart from null when the
// pose smoother releases or re-acquires. Pure, state in → state out, like
// smoothLinePose.

// Fold one read into the accumulator state (or start from prev = null). Each cell
// keeps weighted RGB sums; a low-confidence read (small weight) barely moves a
// well-established mean.
export function accumulateStickerColors(prev, read, opts = {}) {
  const next = { faces: {} };
  if (prev) {
    for (const [key, f] of Object.entries(prev.faces)) {
      next.faces[key] = { k: f.k, s: f.s, cells: f.cells.map((a) => ({ ...a })) };
    }
  }
  if (read) {
    for (const f of read.faces) {
      const key = `${f.k}${f.s > 0 ? '+' : '-'}`;
      const acc = next.faces[key] ??= {
        k: f.k, s: f.s,
        cells: Array.from({ length: 9 }, () => ({ wr: 0, wg: 0, wb: 0, w: 0, n: 0 })),
      };
      f.cells.forEach((c, i) => {
        if (!c.rgb || !(c.weight > 0)) return;
        const a = acc.cells[i];
        a.wr += c.weight * c.rgb.r;
        a.wg += c.weight * c.rgb.g;
        a.wb += c.weight * c.rgb.b;
        a.w += c.weight;
        a.n++;
      });
    }
  }
  return next;
}

// The accumulated weighted-mean colour per cell, in the same faces/cells shape as a
// read (rgb null where nothing has been read yet).
export function accumulatedColors(state) {
  if (!state) return null;
  return {
    faces: Object.values(state.faces).map((f) => ({
      k: f.k, s: f.s,
      cells: f.cells.map((a) => ({
        rgb: a.w > 0 ? { r: a.wr / a.w, g: a.wg / a.w, b: a.wb / a.w } : null,
        weight: a.w, n: a.n,
      })),
    })),
  };
}

// What the live overlay shows: the CURRENT read's faces and pixel positions (so
// swatches track the cube and vanish with it) with each cell's colour replaced by
// the steadier accumulated mean where one exists.
export function overlayColors(read, state) {
  if (!read) return null;
  if (!state) return read.faces;
  return read.faces.map((f) => {
    const acc = state.faces[`${f.k}${f.s > 0 ? '+' : '-'}`];
    return {
      ...f,
      cells: f.cells.map((c, i) => {
        const a = acc && acc.cells[i];
        return a && a.w > 0 ? { ...c, rgb: { r: a.wr / a.w, g: a.wg / a.w, b: a.wb / a.w } } : c;
      }),
    };
  });
}

// Verifies src/group.js (findFaceGrids) groups a noisy quad-centre cloud into cube
// faces: recovers the 3x3 lattice, labels each sticker with a consistent (row,col),
// and rejects clutter that doesn't sit on a grid. Ground truth is synthetic — an
// affine grid, a real perspective projection (via pose.js), and three faces at once
// — so this runs offline. What it canNOT check is detection on real pixels.
//
// Run: npm test

import { findFaceGrids, fitFaceGrid, orientFace } from '../src/group.js';
import { estimateIntrinsics, project } from '../src/pose.js';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name + (cond ? '' : `  ${extra}`));
}

// Deterministic pseudo-random jitter so the test is stable.
let seed = 1;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const jit = (a) => a + (rnd() - 0.5) * 1.5;

function quad(center, tag, area = 100) {
  const s = Math.sqrt(area) / 2;
  return {
    center, tag, area,
    corners: [
      { x: center.x - s, y: center.y - s }, { x: center.x + s, y: center.y - s },
      { x: center.x + s, y: center.y + s }, { x: center.x - s, y: center.y + s },
    ],
  };
}
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; };

// Build quads for a grid of centres, sizing each ~ the local grid pitch — as on a
// real cube, where a sticker fills most of its cell (the grouping relies on that).
function gridQuads(centers, tag) {
  const nn = centers.map((p, i) =>
    Math.min(...centers.filter((_, j) => j !== i).map((q) => Math.hypot(p.x - q.x, p.y - q.y))));
  const pitch = [...nn].sort((a, b) => a - b)[nn.length >> 1];
  return centers.map((c) => quad(c, tag, (pitch * 0.85) ** 2));
}

const cellsCoverFullGrid = (cells) => {
  const keys = new Set(cells.map((c) => `${c.row},${c.col}`));
  if (keys.size !== 9) return false;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) if (!keys.has(`${r},${c}`)) return false;
  return true;
};

// Are the three quads in each labelled row (and column) collinear? Proves the
// (row,col) labelling is a genuine grid, not a scramble. (Affine grids only.)
function labellingIsGridConsistent(cells) {
  const at = (r, c) => cells.find((x) => x.row === r && x.col === c)?.quad.center;
  const collinear = (p, q, s) => Math.abs((q.x - p.x) * (s.y - p.y) - (q.y - p.y) * (s.x - p.x)) < 1e-6;
  for (let r = 0; r < 3; r++) if (!collinear(at(r, 0), at(r, 1), at(r, 2))) return false;
  for (let c = 0; c < 3; c++) if (!collinear(at(0, c), at(1, c), at(2, c))) return false;
  return true;
}

// --- 1. affine face + clutter ----------------------------------------------

(() => {
  const origin = { x: 300, y: 200 }, du = { x: 42, y: 6 }, dv = { x: -7, y: 40 };
  const centers = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    centers.push({ x: origin.x + c * du.x + r * dv.x, y: origin.y + c * du.y + r * dv.y });
  const face = gridQuads(centers, 'face');
  const clutter = [
    quad({ x: 700, y: 90 }, 'clutter', 1200), quad({ x: 60, y: 480 }, 'clutter', 1300),
    quad({ x: 720, y: 520 }, 'clutter', 1100), quad({ x: 120, y: 60 }, 'clutter', 1250),
  ];
  const faces = findFaceGrids(shuffle([...face, ...clutter]));
  check('affine: exactly one face found', faces.length === 1, `got ${faces.length}`);
  const cells = faces[0]?.cells ?? [];
  check('affine: face has 9 cells', cells.length === 9, `got ${cells.length}`);
  check('affine: cells cover a full 3x3', cellsCoverFullGrid(cells));
  check('affine: only real stickers grouped (clutter excluded)', cells.every((c) => c.quad.tag === 'face'));
  check('affine: (row,col) labelling is grid-consistent', labellingIsGridConsistent(cells));
})();

// --- 2. real perspective projection + sub-pixel noise -----------------------

(() => {
  const K = estimateIntrinsics(1280, 720, 60);
  const pose = {
    R: [[0.86, -0.10, 0.50], [0.16, 0.98, -0.07], [-0.48, 0.14, 0.86]], // a moderate oblique view
    t: [0.6, -0.3, 9],
  };
  const centers = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const [x, y] = project(K, pose, [c - 1, r - 1, 0]); // sticker centres on the Z=0 face
    centers.push({ x: jit(x), y: jit(y) });
  }
  const face = gridQuads(centers, 'face');
  const clutter = [quad({ x: 120, y: 110 }, 'clutter', 12000), quad({ x: 1170, y: 650 }, 'clutter', 13000)];
  const faces = findFaceGrids(shuffle([...face, ...clutter]));
  check('perspective: one face found', faces.length === 1, `got ${faces.length}`);
  const cells = faces[0]?.cells ?? [];
  check('perspective: recovers >=8 of 9 stickers', cells.length >= 8, `got ${cells.length}`);
  check('perspective: no clutter grouped', cells.every((c) => c.quad.tag === 'face'));
})();

// --- 3. clutter only: nothing spurious -------------------------------------

(() => {
  const clutter = [];
  for (let i = 0; i < 8; i++) clutter.push(quad({ x: rnd() * 1280, y: rnd() * 720 }, 'clutter', 800 + rnd() * 600));
  const faces = findFaceGrids(clutter);
  const big = faces.filter((f) => f.cells.length >= 4);
  check('clutter only: no 4+ sticker face hallucinated', big.length === 0, `got ${big.length}`);
})();

// --- 4. corner-on: three faces at once -------------------------------------

(() => {
  const bases = [
    { o: { x: 360, y: 250 }, du: { x: 40, y: 8 }, dv: { x: -6, y: 38 } },
    { o: { x: 620, y: 300 }, du: { x: 36, y: -10 }, dv: { x: 8, y: 40 } },
    { o: { x: 470, y: 470 }, du: { x: 42, y: 4 }, dv: { x: -10, y: 36 } },
  ];
  const all = [];
  bases.forEach((b, f) => {
    const centers = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      centers.push({ x: b.o.x + c * b.du.x + r * b.dv.x, y: b.o.y + c * b.du.y + r * b.dv.y });
    all.push(...gridQuads(centers, `f${f}`));
  });
  const faces = findFaceGrids(shuffle(all));
  check('corner-on: three faces found', faces.length === 3, `got ${faces.length}`);
  check('corner-on: each face is a full 3x3', faces.every((f) => cellsCoverFullGrid(f.cells)));
  check('corner-on: each face groups one source face only',
    faces.every((f) => new Set(f.cells.map((c) => c.quad.tag)).size === 1));
})();

// --- 5. fitFaceGrid fills in the stickers detection missed -----------------

(() => {
  const K = estimateIntrinsics(1280, 720, 60);
  const pose = { R: [[0.9, -0.08, 0.43], [0.12, 0.99, -0.06], [-0.42, 0.11, 0.90]], t: [0.4, -0.2, 8.5] };
  const truth = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const [x, y] = project(K, pose, [c - 1, r - 1, 0]);
    truth.push({ row: r, col: c, x, y });
  }
  // Keep 6 of 9 (drop centre + two corners) — a plausible partial detection.
  const dropped = new Set(['1,1', '0,0', '2,2']);
  const kept = truth.filter((t) => !dropped.has(`${t.row},${t.col}`))
    .map((t) => ({ row: t.row, col: t.col, quad: quad({ x: t.x, y: t.y }, 'face', 9000) }));

  const fit = fitFaceGrid({ cells: kept });
  check('fit: returns a fitted grid from 6 cells', !!fit);
  check('fit: projects all 9 cells', fit?.cells.length === 9, `got ${fit?.cells.length}`);
  let worst = 0;
  for (const t of truth) {
    const p = fit.cells.find((c) => c.row === t.row && c.col === t.col);
    worst = Math.max(worst, Math.hypot(p.x - t.x, p.y - t.y));
  }
  check('fit: every projected cell (incl. the 3 missing) lands on the true sticker', worst < 0.5, `worst=${worst.toFixed(3)}px`);
  check('fit: dropped cells flagged not-detected, kept flagged detected', fit.cells.every((c) =>
    c.detected === !dropped.has(`${c.row},${c.col}`)));
})();

// --- 6. degenerate (near-collinear) cell sets are rejected ------------------

(() => {
  // Five cells whose centres lie almost on a line (only one row of the face really
  // detected) — a homography here would fling the projected grid off the cube.
  const line = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]].map(([col, row], i) => ({
    row, col, quad: quad({ x: 200 + i * 50, y: 300 + i * 0.4 }, 'face', 1800),
  }));
  check('degenerate: near-collinear cells are rejected', fitFaceGrid({ cells: line }) === null);

  // A genuine 2x2 block fits fine.
  const block = [[0, 0], [1, 0], [0, 1], [1, 1]].map(([col, row]) => ({
    row, col, quad: quad({ x: 200 + col * 50, y: 300 + row * 48 }, 'face', 1800),
  }));
  check('degenerate: a real 2x2 block still fits', fitFaceGrid({ cells: block }) !== null);
})();

// --- 7. orientFace canonicalizes the cell labelling -------------------------

(() => {
  // Axis-aligned geometry but deliberately flipped labels (col increases LEFT,
  // rows swapped). orientFace should restore col->+x, row->+y.
  const origin = { x: 100, y: 100 }, step = 40, cells = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    cells.push({ row: 2 - r, col: 2 - c, quad: quad({ x: origin.x + c * step, y: origin.y + r * step }, 'face', 1200) });
  const o = orientFace(cells);
  const at = (r, c) => o.find((x) => x.row === r && x.col === c).quad.center;
  let ok = true;
  for (let r = 0; r < 3; r++) if (!(at(r, 0).x < at(r, 1).x && at(r, 1).x < at(r, 2).x)) ok = false;
  for (let c = 0; c < 3; c++) if (!(at(0, c).y < at(1, c).y && at(1, c).y < at(2, c).y)) ok = false;
  check('orientFace: canonicalizes col->+x, row->+y from a flipped labelling', ok);
})();

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

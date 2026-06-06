// Group a noisy cloud of detected sticker quads into cube faces.
//
// Detection (detect.js) returns far more than a clean 3x3: real frames give maybe
// half a face's stickers plus false positives (background, fingers, merged blocks).
// findFaceGrids finds affine 3x3 lattices among the quad CENTRES by RANSAC — which
// is the false-positive rejection (clutter doesn't sit on a grid) and assigns each
// inlier a (row, col). A face's assigned cells then drive a precise homography
// (pose.js) -> 6-DoF, and projecting the cube model fills in the stickers detection
// missed. An affine lattice is only an approximation of the true perspective grid,
// but it's enough to GROUP and label; the homography that follows is full
// perspective.
//
// Offline-tested against synthetic projected cubes (test/group.test.mjs).

import { homographyDLT, applyHomography } from './pose.js';

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const len = (a) => Math.hypot(a.x, a.y);

// Lattice coordinates of p in the basis (du, dv) anchored at origin: solve
// [du dv][i j]^T = p - origin. Returns null if the basis is degenerate.
function latticeCoords(p, origin, du, dv) {
  const det = du.x * dv.y - dv.x * du.y;
  if (Math.abs(det) < 1e-9) return null;
  const dx = p.x - origin.x, dy = p.y - origin.y;
  return { i: (dv.y * dx - dv.x * dy) / det, j: (-du.y * dx + du.x * dy) / det };
}

// Ratio of the smaller to larger eigenvalue of a point set's covariance:
// ~0 = points collinear, ~1 = isotropic. Used to reject degenerate face fits.
function spread2D(points) {
  let mx = 0, my = 0;
  for (const p of points) { mx += p.x; my += p.y; }
  mx /= points.length; my /= points.length;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of points) { const dx = p.x - mx, dy = p.y - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + disc;
  return l1 > 0 ? (tr / 2 - disc) / l1 : 0;
}

const areaRatio = (a, b) => {
  const r = (a.area ?? 1) / (b.area ?? 1);
  return r < 1 ? 1 / r : r;
};

// Best 3x3 lattice found among `quads`, or null. Returns { cells, score } where
// cells = [{ row, col, quad }] with row,col in 0..2 (one quad per cell).
function bestLattice(quads, o) {
  let best = null;

  for (let s = 0; s < quads.length; s++) {
    const seed = quads[s];
    // Adjacent stickers are the nearest similar-sized neighbours; the basis steps
    // come from pairs of them.
    const near = quads
      .map((q, idx) => ({ idx, d: dist(seed.center, q.center) }))
      .filter((c) => c.idx !== s && areaRatio(quads[c.idx], seed) < o.sizeRatio)
      .sort((a, b) => a.d - b.d)
      .slice(0, o.neighbors);

    for (let a = 0; a < near.length; a++) {
      for (let b = a + 1; b < near.length; b++) {
        const du = sub(quads[near[a].idx].center, seed.center);
        const dv = sub(quads[near[b].idx].center, seed.center);
        const lu = len(du), lv = len(dv);
        const cos = (du.x * dv.x + du.y * dv.y) / (lu * lv || 1);
        if (Math.abs(cos) > o.maxBasisCos) continue; // near-parallel: not a grid basis

        // A real cube's grid step ≈ sticker size, and the grid is roughly square.
        // This is the main clutter rejector: random points have a lattice step
        // unrelated to (much larger than) their quad size.
        const pitch = Math.sqrt(seed.area || 1);
        if (lu < o.pitchMin * pitch || lu > o.pitchMax * pitch) continue;
        if (lv < o.pitchMin * pitch || lv > o.pitchMax * pitch) continue;
        if (Math.max(lu, lv) / Math.min(lu, lv) > o.maxPitchRatio) continue;

        const hit = latticeInliers(quads, seed.center, du, dv, o);
        if (hit && (!best || hit.score > best.score)) best = hit;
      }
    }
  }
  return best;
}

// Quads whose lattice coords are near-integer, reduced to the densest 3x3 window
// with at most one quad per cell.
function latticeInliers(quads, origin, du, dv, o) {
  const onLattice = [];
  for (let k = 0; k < quads.length; k++) {
    const lc = latticeCoords(quads[k].center, origin, du, dv);
    if (!lc) return null;
    const ri = Math.round(lc.i), rj = Math.round(lc.j);
    const err = Math.hypot(lc.i - ri, lc.j - rj);
    if (Math.abs(lc.i - ri) < o.latticeTol && Math.abs(lc.j - rj) < o.latticeTol) {
      onLattice.push({ k, i: ri, j: rj, err });
    }
  }
  if (onLattice.length < o.minInliers) return null;

  // Slide a 3x3 window to the position covering the most inliers.
  let win = null;
  const is = onLattice.map((p) => p.i), js = onLattice.map((p) => p.j);
  for (let i0 = Math.min(...is) - 2; i0 <= Math.max(...is); i0++) {
    for (let j0 = Math.min(...js) - 2; j0 <= Math.max(...js); j0++) {
      const inWin = onLattice.filter((p) => p.i >= i0 && p.i <= i0 + 2 && p.j >= j0 && p.j <= j0 + 2);
      if (!win || inWin.length > win.list.length) win = { i0, j0, list: inWin };
    }
  }

  // One quad per cell: keep the lowest-error inlier in each (col,row).
  const byCell = new Map();
  for (const p of win.list) {
    const col = p.i - win.i0, row = p.j - win.j0, key = `${row},${col}`;
    const prev = byCell.get(key);
    if (!prev || p.err < prev.err) byCell.set(key, { row, col, err: p.err, quad: quads[p.k] });
  }
  const cells = [...byCell.values()].map(({ row, col, quad }) => ({ row, col, quad }));
  if (cells.length < o.minInliers) return null;
  return { cells, score: cells.length };
}

// Find up to maxFaces 3x3 grids in the quad cloud. Each returned face is
// { cells: [{ row, col, quad }] }. Quads claimed by one face aren't reused.
export function findFaceGrids(quads, opts = {}) {
  const o = {
    latticeTol: 0.22,  // max |fractional lattice coord| to count as on-grid
    minInliers: 4,     // a face needs at least this many detected stickers
    maxFaces: 3,       // corner-on shows three
    neighbors: 4,      // nearest neighbours considered for basis pairs
    sizeRatio: 2.5,    // a face's stickers are within this area ratio of each other
    maxBasisCos: 0.7,  // reject near-parallel basis pairs (not a grid)
    pitchMin: 0.55,    // grid step must be within [min,max] x sqrt(sticker area):
    pitchMax: 3.0,     //   a real step ≈ sticker size, clutter's is far larger
    maxPitchRatio: 2.5, // the two grid steps are comparable (roughly square)
    ...opts,
  };

  let pool = quads.map((q, idx) => ({ ...q, _idx: idx }));
  const faces = [];
  for (let f = 0; f < o.maxFaces; f++) {
    const best = bestLattice(pool, o);
    if (!best) break;
    faces.push({ cells: best.cells.map(({ row, col, quad }) => ({ row, col, quad })) });
    const used = new Set(best.cells.map((c) => c.quad._idx));
    pool = pool.filter((q) => !used.has(q._idx));
    if (pool.length < o.minInliers) break;
  }
  return faces;
}

// Relabel a face's cells to a canonical image orientation: column axis pointing
// mostly +x, row axis mostly +y. Grouping's lattice basis is arbitrary (it can come
// out transposed/flipped frame to frame), so without this the projected grid's
// point order would flip between frames and break temporal smoothing.
export function orientFace(cells) {
  const at = (r, c) => cells.find((x) => x.row === r && x.col === c)?.quad.center;
  let cx = 0, cy = 0, cn = 0, rx = 0, ry = 0, rn = 0;
  for (const cell of cells) {
    const a = cell.quad.center;
    const right = at(cell.row, cell.col + 1); if (right) { cx += right.x - a.x; cy += right.y - a.y; cn++; }
    const down = at(cell.row + 1, cell.col); if (down) { rx += down.x - a.x; ry += down.y - a.y; rn++; }
  }
  if (!cn || !rn) return cells; // not enough adjacency to judge orientation
  let col = { x: cx / cn, y: cy / cn }, row = { x: rx / rn, y: ry / rn };
  let transpose = false;
  if (Math.abs(col.x) < Math.abs(row.x)) { transpose = true; [col, row] = [row, col]; }
  const flipC = col.x < 0, flipR = row.y < 0;
  return cells.map(({ row: r, col: c, quad }) => {
    let nr = r, nc = c;
    if (transpose) { const t = nr; nr = nc; nc = t; }
    if (flipC) nc = 2 - nc;
    if (flipR) nr = 2 - nr;
    return { row: nr, col: nc, quad };
  });
}

// Fit a homography from a face's labelled cells (grid (col,row) -> image) and
// project the FULL 3x3, filling in the stickers detection missed. This is the
// payoff of grouping: a partial, jittery detection becomes a complete face whose
// every sticker has a known image position to sample and overlay. Returns
//   { H, cells: [{ row, col, x, y, detected }*9], outline: [4 corners] }
// or null if the cells don't span a 2D grid (homography under-determined).
export function fitFaceGrid(face, opts = {}) {
  const o = { minSpread: 0.05, ...opts };
  const cells = orientFace(face.cells); // canonical labels -> stable projection order
  const cols = new Set(cells.map((c) => c.col)), rows = new Set(cells.map((c) => c.row));
  if (cells.length < 4 || cols.size < 2 || rows.size < 2) return null;
  // Near-collinear cells (a single detected row/line) make the homography
  // under-determined, so the projected grid flies off the cube. Require the cell
  // centres to span 2D: the covariance's smaller/larger eigenvalue ratio.
  if (spread2D(cells.map((c) => c.quad.center)) < o.minSpread) return null;

  const H = homographyDLT(cells.map((c) => ({ X: [c.col, c.row], u: [c.quad.center.x, c.quad.center.y] })));
  const detected = new Set(cells.map((c) => `${c.row},${c.col}`));
  const projected = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const [x, y] = applyHomography(H, [c, r]);
      projected.push({ row: r, col: c, x, y, detected: detected.has(`${r},${c}`) });
    }
  }
  const corner = (c, r) => { const [x, y] = applyHomography(H, [c, r]); return { x, y }; };
  const outline = [corner(-0.5, -0.5), corner(2.5, -0.5), corner(2.5, 2.5), corner(-0.5, 2.5)];
  return { H, cells: projected, outline };
}

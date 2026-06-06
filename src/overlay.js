// Canvas overlays: the scan guide grid and the AR move arrows.
//
// Move-direction geometry (front face toward camera, up face on top), derived
// from rotating the standard cube frame:
//   U  -> top row shifts LEFT      U' -> RIGHT
//   D  -> bottom row shifts RIGHT  D' -> LEFT
//   R  -> right column UP          R' -> DOWN
//   L  -> left column DOWN         L' -> UP
//   F  -> whole face clockwise     F' -> counter-clockwise
//   B  -> appears counter-clockwise from the front  B' -> clockwise
// A "2" suffix means the same direction, performed twice.

import { FACE_DISPLAY } from './colors.js';

const ARROWS = {
  U:   { kind: 'row', i: 0, dir: 'left' },
  "U'":{ kind: 'row', i: 0, dir: 'right' },
  D:   { kind: 'row', i: 2, dir: 'right' },
  "D'":{ kind: 'row', i: 2, dir: 'left' },
  R:   { kind: 'col', i: 2, dir: 'up' },
  "R'":{ kind: 'col', i: 2, dir: 'down' },
  L:   { kind: 'col', i: 0, dir: 'down' },
  "L'":{ kind: 'col', i: 0, dir: 'up' },
  F:   { kind: 'rot', dir: 'cw' },
  "F'":{ kind: 'rot', dir: 'ccw' },
  B:   { kind: 'rot', dir: 'ccw', back: true },
  "B'":{ kind: 'rot', dir: 'cw', back: true },
};

export function parseMove(tok) {
  const face = tok[0];
  const mod = tok.slice(1); // '', "'", or '2'
  const key = mod === "'" ? `${face}'` : face;
  return { face, token: tok, double: mod === '2', spec: ARROWS[key] };
}

// --- low-level drawing helpers ---------------------------------------------

function arrowhead(ctx, x, y, angle, size) {
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(angle - 0.4), y - size * Math.sin(angle - 0.4));
  ctx.lineTo(x - size * Math.cos(angle + 0.4), y - size * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fill();
}

function straightArrow(ctx, x1, y1, x2, y2, w, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = w;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  arrowhead(ctx, x2, y2, Math.atan2(y2 - y1, x2 - x1), w * 2.4);
  ctx.restore();
}

function arcArrow(ctx, cx, cy, radius, ccw, w, color, dashed) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = w;
  ctx.lineCap = 'round';
  if (dashed) ctx.setLineDash([12, 10]);
  const a0 = -Math.PI * 0.85;
  const a1 = Math.PI * 0.85;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, ccw ? a1 : a0, ccw ? a0 : a1, ccw);
  ctx.stroke();
  ctx.setLineDash([]);
  // Arrowhead tangent to the arc at the end angle.
  const end = ccw ? a0 : a1;
  const hx = cx + radius * Math.cos(end);
  const hy = cy + radius * Math.sin(end);
  const tangent = end + (ccw ? -Math.PI / 2 : Math.PI / 2);
  arrowhead(ctx, hx, hy, tangent, w * 2.6);
  ctx.restore();
}

// --- public API ------------------------------------------------------------

// The alignment grid shown while scanning.
export function drawGuide(ctx, region, active) {
  const { x, y, side, cell } = region;
  ctx.save();
  ctx.strokeStyle = active ? 'rgba(79,140,255,0.95)' : 'rgba(255,255,255,0.65)';
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, side, side);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  for (let k = 1; k < 3; k++) {
    ctx.beginPath();
    ctx.moveTo(x + cell * k, y);
    ctx.lineTo(x + cell * k, y + side);
    ctx.moveTo(x, y + cell * k);
    ctx.lineTo(x + side, y + cell * k);
    ctx.stroke();
  }
  ctx.restore();
}

// The corner-on alignment guide: the projected cube silhouette split into three
// rhombus faces, each with a 3x3 grid, that the user lines a cube corner up with.
// Geometry comes from the projected `scene` (computeCornerRegion), so it tracks
// the perspective slider. Each face's rhombus outline draws the silhouette edges
// and the near spokes; the grid lines are drawn fainter inside.
export function drawCornerGuide(ctx, scene, active) {
  ctx.save();

  ctx.strokeStyle = active ? 'rgba(79,140,255,0.95)' : 'rgba(255,255,255,0.65)';
  ctx.lineWidth = 3;
  for (const face of scene.faces) {
    ctx.beginPath();
    face.quad.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.stroke();
  }

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.beginPath();
  for (const face of scene.faces) {
    for (const [a, b] of face.grid) {
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

// Show the classified result of a scanned face as 9 colored cells.
export function drawClassified(ctx, region, labels) {
  const { x, y, cell } = region;
  ctx.save();
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      ctx.fillStyle = FACE_DISPLAY[labels[r * 3 + c]] ?? '#888';
      const pad = cell * 0.12;
      ctx.fillRect(x + cell * c + pad, y + cell * r + pad, cell - 2 * pad, cell - 2 * pad);
    }
  }
  ctx.restore();
}

// Debug overlay for the auto-scan detector: outline each detected sticker quad
// and dot its centre. Geometry only (no canvas text — it would read backwards
// under the preview mirror); any counts/readouts belong in the HTML chrome.
export function drawDetections(ctx, quads) {
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(0,240,140,0.95)';
  ctx.fillStyle = 'rgba(0,240,140,0.95)';
  for (const q of quads) {
    ctx.beginPath();
    q.corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(q.center.x, q.center.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// Debug overlay for grouping: draw each recovered cube face's cells in a distinct
// colour and connect adjacent grid cells, so the 3x3 lattices found among the
// detected quads (and the rejection of clutter) are visible live. Geometry only.
const FACE_COLORS = ['rgba(0,200,255,0.95)', 'rgba(255,90,200,0.95)', 'rgba(255,210,0,0.95)'];
export function drawGrids(ctx, faces) {
  ctx.save();
  faces.forEach((face, fi) => {
    const color = FACE_COLORS[fi % FACE_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    for (const cell of face.cells) {
      const q = cell.quad;
      ctx.beginPath();
      q.corners.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
      ctx.closePath();
      ctx.stroke();
    }
    // Connect each cell to its right/down grid neighbour to show the lattice.
    const at = (r, c) => face.cells.find((x) => x.row === r && x.col === c)?.quad.center;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (const cell of face.cells) {
      const a = cell.quad.center;
      const right = at(cell.row, cell.col + 1), down = at(cell.row + 1, cell.col);
      if (right) { ctx.moveTo(a.x, a.y); ctx.lineTo(right.x, right.y); }
      if (down) { ctx.moveTo(a.x, a.y); ctx.lineTo(down.x, down.y); }
    }
    ctx.stroke();
  });
  ctx.restore();
}

// Draw fitted faces: the full 3x3 projected from each face's homography, so the
// stickers detection missed are filled in. Outline per-face colour; each cell a
// solid dot if it was actually detected, a hollow ring if projection-filled.
export function drawFittedFaces(ctx, fits) {
  ctx.save();
  fits.forEach((fit, fi) => {
    const color = FACE_COLORS[fi % FACE_COLORS.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    fit.outline.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.closePath();
    ctx.stroke();
    for (const c of fit.cells) {
      ctx.beginPath();
      ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
      if (c.detected) ctx.fill();
      else { ctx.lineWidth = 2; ctx.stroke(); }
    }
  });
  ctx.restore();
}

// Draw a recovered cube pose: each visible face's projected 3x3 (cells in row-major
// order) as a connected grid + dots, per-face colour. Geometry only.
export function drawCube(ctx, faces) {
  ctx.save();
  faces.forEach((f, fi) => {
    const color = FACE_COLORS[fi % FACE_COLORS.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    const at = (r, c) => f.cells[r * 3 + c];
    ctx.beginPath();
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const p = at(r, c);
      if (c < 2) { const q = at(r, c + 1); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); }
      if (r < 2) { const q = at(r + 1, c); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); }
    }
    ctx.stroke();
    for (const p of f.cells) { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2); ctx.fill(); }
  });
  ctx.restore();
}

// Draw the AR arrow for a single move on the front-facing grid.
export function drawMove(ctx, region, token) {
  const { x, y, side, cell } = region;
  const { spec } = parseMove(token);
  if (!spec) return;

  const color = spec.back ? '#ffd60a' : '#4f8cff';
  const w = Math.max(8, side * 0.045);
  const inset = cell * 0.55;

  ctx.save();

  if (spec.kind === 'row') {
    const cy = y + cell * (spec.i + 0.5);
    const left = x + inset, right = x + side - inset;
    // Translucent band over the affected row.
    ctx.fillStyle = 'rgba(79,140,255,0.18)';
    ctx.fillRect(x, y + cell * spec.i, side, cell);
    if (spec.dir === 'right') straightArrow(ctx, left, cy, right, cy, w, color);
    else straightArrow(ctx, right, cy, left, cy, w, color);
  } else if (spec.kind === 'col') {
    const cx = x + cell * (spec.i + 0.5);
    const top = y + inset, bottom = y + side - inset;
    ctx.fillStyle = 'rgba(79,140,255,0.18)';
    ctx.fillRect(x + cell * spec.i, y, cell, side);
    if (spec.dir === 'up') straightArrow(ctx, cx, bottom, cx, top, w, color);
    else straightArrow(ctx, cx, top, cx, bottom, w, color);
  } else {
    // Whole-face rotation.
    ctx.fillStyle = spec.back ? 'rgba(255,214,10,0.14)' : 'rgba(79,140,255,0.14)';
    ctx.fillRect(x, y, side, side);
    arcArrow(ctx, x + side / 2, y + side / 2, side * 0.32,
      spec.dir === 'ccw', w, color, spec.back);
  }

  // The move label is rendered in the (un-mirrored) HTML status bar rather than
  // on the canvas, so it stays readable when the preview is mirrored.
  ctx.restore();
}

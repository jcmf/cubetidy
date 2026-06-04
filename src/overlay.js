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

// A bowed arrow from (x1,y1) to (x2,y2); `bow` offsets the control point
// perpendicular to the line (0 = straight).
function sweepArrow(ctx, x1, y1, x2, y2, bow, w, color) {
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const px = -dy / len, py = dx / len; // unit perpendicular
  const cxp = mx + px * bow, cyp = my + py * bow;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = w;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.quadraticCurveTo(cxp, cyp, x2, y2);
  ctx.stroke();
  arrowhead(ctx, x2, y2, Math.atan2(y2 - cyp, x2 - cxp), w * 2.4);
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

// Draw the reorientation indicator for a scan step. `motion` is one of:
//   'front'      - no rotation (just frame a face)
//   'spin'       - spin about the vertical axis, bringing the right face front
//   'tiltTop'    - tilt the top toward the camera (front face rolls down)
//   'tiltBottom' - tilt the bottom toward the camera (front face rolls up)
// Drawn in true canvas coordinates, so the CSS mirror flips it together with the
// cube — "make the face follow the arrow" stays correct in either mirror state.
export function drawScanIndicator(ctx, region, motion) {
  if (!motion || motion === 'front') return;
  const { x, y, side } = region;
  const cx = x + side / 2, cy = y + side / 2;
  const color = '#ffcc00';
  const w = Math.max(10, side * 0.06);

  ctx.save();
  ctx.shadowColor = 'rgba(0,0,0,0.6)';
  ctx.shadowBlur = 8;

  if (motion === 'spin') {
    // Front face slides toward image-left; bowed arrow across the top band.
    const ay = y + side * 0.16;
    const half = side * 0.3;
    sweepArrow(ctx, cx + half, ay, cx - half, ay, side * 0.13, w, color);
  } else if (motion === 'tiltTop') {
    // Front face rolls downward; arrow in the upper area pointing down.
    sweepArrow(ctx, cx, y + side * 0.14, cx, y + side * 0.5, 0, w, color);
  } else if (motion === 'tiltBottom') {
    // Front face rolls upward; arrow in the lower area pointing up.
    sweepArrow(ctx, cx, y + side * 0.86, cx, y + side * 0.5, 0, w, color);
  }

  ctx.restore();
}

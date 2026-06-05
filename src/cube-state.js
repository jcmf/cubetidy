// Assembling the scanned faces into a Kociemba facelet string and validating it.
//
// The scan is guided so each captured 3x3 (row-major, as the camera sees it)
// maps DIRECTLY onto that face's facelet positions. See SCAN_STEPS for the exact
// physical orientation the user is asked to hold for each capture.

import { classifyFaces } from './colors.js';

// Order in which faces are captured, and the holding instruction for each.
// Letters are Kociemba face letters (U/R/F/D/L/B).
export const SCAN_STEPS = [
  { face: 'F', title: 'FRONT', motion: 'front',
    hint: 'Point <b>any</b> face at the camera and fill the grid. This is your reference <b>front</b>.' },
  { face: 'R', title: 'RIGHT', motion: 'spin',
    hint: 'Spin the cube to bring its <b>right</b> side to the front — follow the arrow. Keep the same face on top.' },
  { face: 'B', title: 'BACK', motion: 'spin',
    hint: 'Spin the same way again to bring the <b>back</b> face to the front.' },
  { face: 'L', title: 'LEFT', motion: 'spin',
    hint: 'Spin once more to bring the <b>left</b> side to the front.' },
  { face: 'U', title: 'UP', motion: 'tiltTop',
    hint: 'Return to the front, then tilt the <b>top</b> toward the camera — follow the arrow.' },
  { face: 'D', title: 'DOWN', motion: 'tiltBottom',
    hint: 'Return to the front, then tilt the <b>bottom</b> toward the camera — follow the arrow.' },
];

// Kociemba facelet string order.
const FACELET_ORDER = ['U', 'R', 'F', 'D', 'L', 'B'];

// Build the 54-char facelet string from captured faces.
// `faces` maps face letter -> 9 {r,g,b} samples (row-major as captured).
// `conflicts` is the classifier's residual-ambiguity count (see classifyFaces).
export function toFaceletString(faces) {
  const { labels, counts, conflicts } = classifyFaces(faces);
  let out = '';
  for (const letter of FACELET_ORDER) out += labels[letter].join('');
  return { facelets: out, counts, conflicts };
}

// Average each facelet's RGB across one or more scan passes. The capture geometry
// is deterministic, so facelet i of face F means the same sticker every pass —
// no registration needed. Folding in extra passes from different angles rides out
// per-sticker lighting/glare (the red-vs-orange boundary) before classification.
export function aggregateFaces(passes) {
  const out = {};
  for (const letter of FACELET_ORDER) {
    out[letter] = [];
    for (let i = 0; i < 9; i++) {
      let r = 0, g = 0, b = 0, n = 0;
      for (const p of passes) {
        const s = p[letter]?.[i];
        if (s) { r += s.r; g += s.g; b += s.b; n++; }
      }
      out[letter].push({ r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) });
    }
  }
  return out;
}

// Sanity-check before handing to the solver. Returns { ok, error }.
export function validate(facelets, counts) {
  for (const letter of FACELET_ORDER) {
    if ((counts[letter] ?? 0) !== 9) {
      return {
        ok: false,
        error: `Color counts are off (${letter}: ${counts[letter] ?? 0}/9). ` +
          `Two stickers were likely misread — re-scan in even lighting.`,
      };
    }
  }
  // Centers must each be their own letter (guaranteed by calibration, but verify).
  const centers = [4, 13, 22, 31, 40, 49].map((i) => facelets[i]);
  if (centers.join('') !== FACELET_ORDER.join('')) {
    return { ok: false, error: 'Center mismatch — please re-scan.' };
  }
  return { ok: true };
}

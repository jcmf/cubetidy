// Assembling the scanned faces into a Kociemba facelet string and validating it.
//
// The scan is guided so each captured 3x3 (row-major, as the camera sees it)
// maps DIRECTLY onto that face's facelet positions. See SCAN_STEPS for the exact
// physical orientation the user is asked to hold for each capture.

import { buildReferences, classify } from './colors.js';

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
export function toFaceletString(faces) {
  const references = buildReferences(faces);
  let out = '';
  const counts = {};
  for (const letter of FACELET_ORDER) {
    for (const sample of faces[letter]) {
      const { label } = classify(sample, references);
      out += label;
      counts[label] = (counts[label] ?? 0) + 1;
    }
  }
  return { facelets: out, counts };
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

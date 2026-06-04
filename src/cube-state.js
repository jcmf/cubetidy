// Assembling the scanned faces into a Kociemba facelet string and validating it.
//
// The scan is guided so each captured 3x3 (row-major, as the camera sees it)
// maps DIRECTLY onto that face's facelet positions. See SCAN_STEPS for the exact
// physical orientation the user is asked to hold for each capture.

import { buildReferences, classify } from './colors.js';

// Order in which faces are captured, and the holding instruction for each.
// Letters are Kociemba face letters (U/R/F/D/L/B).
export const SCAN_STEPS = [
  { face: 'F', title: 'FRONT',
    hint: 'Point <b>any</b> face at the camera. This is your reference <b>front</b>.' },
  { face: 'R', title: 'RIGHT',
    hint: 'Turn the cube <b>left 90°</b> so the right side now faces the camera. Keep the same face on top.' },
  { face: 'B', title: 'BACK',
    hint: 'Turn <b>left</b> again — the <b>back</b> face now faces the camera.' },
  { face: 'L', title: 'LEFT',
    hint: 'Turn <b>left</b> again — the <b>left</b> side now faces the camera.' },
  { face: 'U', title: 'UP',
    hint: 'Return to the front, then tilt the <b>top</b> of the cube toward the camera.' },
  { face: 'D', title: 'DOWN',
    hint: 'Return to the front, then tilt the <b>bottom</b> of the cube toward the camera.' },
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

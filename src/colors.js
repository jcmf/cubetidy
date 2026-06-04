// Color utilities and facelet classification.
//
// Classification is calibration-based: the six face centers captured during
// scanning define one reference color per face letter (U/R/F/D/L/B). Every
// sticker is then assigned to its nearest reference in CIE-Lab space, which is
// far more robust to lighting than fixed hue thresholds and guarantees that
// each center maps to itself.

export function rgbToLab({ r, g, b }) {
  // sRGB -> linear
  let R = r / 255, G = g / 255, B = b / 255;
  R = R > 0.04045 ? ((R + 0.055) / 1.055) ** 2.4 : R / 12.92;
  G = G > 0.04045 ? ((G + 0.055) / 1.055) ** 2.4 : G / 12.92;
  B = B > 0.04045 ? ((B + 0.055) / 1.055) ** 2.4 : B / 12.92;

  // linear RGB -> XYZ (D65)
  const X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047;
  const Y = (R * 0.2126 + G * 0.7152 + B * 0.0722) / 1.0;
  const Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;

  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);

  return { L: 116 * fy - 16, a: 500 * (fx - fy), b: 200 * (fy - fz) };
}

export function labDistance(p, q) {
  const dL = p.L - q.L, da = p.a - q.a, db = p.b - q.b;
  return Math.sqrt(dL * dL + da * da + db * db);
}

// Build reference colors from the six captured face centers.
// `faces` maps a face letter to an array of 9 {r,g,b} samples (index 4 = center).
export function buildReferences(faces) {
  return Object.entries(faces).map(([label, samples]) => ({
    label,
    lab: rgbToLab(samples[4]),
  }));
}

// Classify a single RGB sample to the nearest reference, returning its face letter.
export function classify(rgb, references) {
  const lab = rgbToLab(rgb);
  let best = references[0], bestD = Infinity;
  for (const ref of references) {
    const d = labDistance(lab, ref.lab);
    if (d < bestD) { bestD = d; best = ref; }
  }
  return { label: best.label, distance: bestD };
}

// Approximate display color for a face letter, for drawing thumbnails/overlays.
export const FACE_DISPLAY = {
  U: '#f5f5f5', // up    (white in the standard scheme)
  D: '#ffd500', // down  (yellow)
  F: '#009b48', // front (green)
  B: '#0046ad', // back  (blue)
  R: '#b71234', // right (red)
  L: '#ff5800', // left  (orange)
};

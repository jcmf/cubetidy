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

// --- balanced, center-anchored classification ------------------------------
//
// Nearest-center alone struggles at the red/orange boundary under warm light. We
// know two strong priors about a real cube: each center's label is fixed, and
// there are exactly nine stickers of every color. classifyFaces uses both — a
// constrained k-means (k=6) over all 54 samples in Lab, with the six centers
// pinned to their own clusters and every cluster forced to exactly nine. The
// clearly-colored stickers claim their slots; the ambiguous ones fall to the only
// label left, which is the correct one. `conflicts` reports how many stickers the
// balance constraint had to pull off their nearest centroid — a residual-
// ambiguity signal the scan loop uses to decide whether to gather another pass.

const LETTERS = ['U', 'R', 'F', 'D', 'L', 'B'];
const PER_FACE = 9;
const KMEANS_ITERS = 6;

const labDist2 = (p, q) => (p.L - q.L) ** 2 + (p.a - q.a) ** 2 + (p.b - q.b) ** 2;

// Assign every point a label with exactly PER_FACE per label, centers pinned to
// their own label, minimizing total squared distance to `centroids`.
function balancedAssign(points, centroids) {
  const cap = Object.fromEntries(LETTERS.map((L) => [L, PER_FACE]));
  const assign = new Array(points.length).fill(null);
  for (let k = 0; k < points.length; k++) {
    if (points[k].center) { assign[k] = points[k].face; cap[points[k].face]--; }
  }
  const cost = (k, L) => labDist2(points[k].lab, centroids[L]);

  // Greedy by decisiveness: the most clear-cut stickers grab their slots first,
  // leaving the genuinely ambiguous ones to fill whatever remains.
  const free = points.map((_, k) => k).filter((k) => !points[k].center);
  const margin = (k) => { const d = LETTERS.map((L) => cost(k, L)).sort((a, b) => a - b); return d[1] - d[0]; };
  free.sort((a, b) => margin(b) - margin(a));
  for (const k of free) {
    let best = null, bd = Infinity;
    for (const L of LETTERS) if (cap[L] > 0 && cost(k, L) < bd) { bd = cost(k, L); best = L; }
    assign[k] = best; cap[best]--;
  }

  // 2-opt polish: swap two stickers' labels whenever it lowers total cost (keeps
  // counts balanced). Settles the greedy result to a local optimum.
  for (let pass = 0; pass < 40; pass++) {
    let moved = false;
    for (let x = 0; x < free.length; x++) {
      for (let y = x + 1; y < free.length; y++) {
        const a = free[x], b = free[y], la = assign[a], lb = assign[b];
        if (la === lb) continue;
        if (cost(a, lb) + cost(b, la) < cost(a, la) + cost(b, lb) - 1e-9) {
          assign[a] = lb; assign[b] = la; moved = true;
        }
      }
    }
    if (!moved) break;
  }
  return assign;
}

// Classify all six faces together. `faces` maps a letter to 9 {r,g,b} samples.
// Returns { labels: {letter: [9 labels]}, counts, conflicts }.
export function classifyFaces(faces) {
  const points = [];
  for (const face of LETTERS) {
    faces[face].forEach((rgb, idx) =>
      points.push({ lab: rgbToLab(rgb), face, idx, center: idx === 4 }));
  }
  const centroids = Object.fromEntries(LETTERS.map((L) => [L, rgbToLab(faces[L][4])]));

  let assign = balancedAssign(points, centroids);
  for (let it = 1; it < KMEANS_ITERS; it++) {
    for (const L of LETTERS) {
      let sL = 0, sa = 0, sb = 0;
      points.forEach((p, k) => { if (assign[k] === L) { sL += p.lab.L; sa += p.lab.a; sb += p.lab.b; } });
      centroids[L] = { L: sL / PER_FACE, a: sa / PER_FACE, b: sb / PER_FACE };
    }
    const next = balancedAssign(points, centroids);
    if (next.every((v, k) => v === assign[k])) { assign = next; break; }
    assign = next;
  }

  const nearest = (lab) => LETTERS.reduce((best, L) =>
    labDist2(lab, centroids[L]) < labDist2(lab, centroids[best]) ? L : best, LETTERS[0]);

  const labels = Object.fromEntries(LETTERS.map((L) => [L, new Array(PER_FACE)]));
  const counts = {};
  let conflicts = 0;
  points.forEach((p, k) => {
    labels[p.face][p.idx] = assign[k];
    counts[assign[k]] = (counts[assign[k]] ?? 0) + 1;
    if (!p.center && assign[k] !== nearest(p.lab)) conflicts++;
  });
  return { labels, counts, conflicts };
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

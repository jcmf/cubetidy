// OpenCV-based cube detection for the auto-scan. BROWSER-ONLY in practice: real
// robustness can only be judged with a camera and a physical cube (`npm test`
// can't cover the pixel layer). The OpenCV `cv` module is passed in rather than
// imported, so this file has no side-effects and an API-level smoke test can feed
// it node's cv (see tools/detect-smoke.mjs).
//
// Stage 1 (here): find candidate sticker quadrilaterals in a frame — convex,
// roughly-square contours of plausible size. Stage 2 (grouping the quads into the
// three faces of a corner-on view and solving a single 6-DoF cube pose) builds on
// this once stage 1 is confirmed against a real cube.
//
// All OpenCV Mats are explicitly deleted: this runs in the render loop, and the
// Emscripten heap is not garbage-collected.

export const DETECT_DEFAULTS = {
  blur: 5,             // Gaussian blur kernel (odd); tames sensor noise pre-edges
  cannyLo: 40,         // Canny hysteresis thresholds
  cannyHi: 120,
  dilateIters: 2,      // dilate edges to close gaps so sticker borders form loops
  approxEps: 0.08,     // approxPolyDP tolerance as a fraction of contour perimeter
  minAreaFrac: 0.0008, // reject quads smaller than this fraction of the frame
  maxAreaFrac: 0.05,   // ...or larger (a whole face/background panel)
  minFill: 0.55,       // contourArea / min-area-rect area: rejects slivers
  maxAspect: 2.2,      // max side ratio of the bounding rotated rect
  medianBand: [0.35, 2.8], // keep quads whose area is within this band of the median
};

// Find candidate sticker quads in an ImageData. Returns an array of
// { corners: [{x,y}*4], center: {x,y}, area } in canvas pixel coordinates.
export function detectStickerQuads(cv, imageData, opts = {}) {
  const o = { ...DETECT_DEFAULTS, ...opts };
  const frameArea = imageData.width * imageData.height;

  const src = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));

  const quads = [];
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(o.blur, o.blur), 0);
    cv.Canny(gray, edges, o.cannyLo, o.cannyHi);
    cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), o.dilateIters);
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const quad = quadFromContour(cv, c, o, frameArea);
      if (quad) quads.push(quad);
      c.delete();
    }
  } finally {
    src.delete(); gray.delete(); edges.delete();
    contours.delete(); hierarchy.delete(); kernel.delete();
  }

  return filterByMedianArea(quads, o.medianBand);
}

// Approve one contour as a sticker quad, or return null. Owns no Mat beyond a
// local approx (deleted before returning).
function quadFromContour(cv, contour, o, frameArea) {
  const area = cv.contourArea(contour);
  if (area < o.minAreaFrac * frameArea || area > o.maxAreaFrac * frameArea) return null;

  const peri = cv.arcLength(contour, true);
  const approx = new cv.Mat();
  let quad = null;
  try {
    cv.approxPolyDP(contour, approx, o.approxEps * peri, true);
    if (approx.rows !== 4 || !cv.isContourConvex(approx)) return null;

    const rect = cv.minAreaRect(approx);
    const w = rect.size.width, h = rect.size.height;
    const aspect = Math.max(w, h) / Math.max(1, Math.min(w, h));
    const fill = area / (w * h || 1);
    if (aspect > o.maxAspect || fill < o.minFill) return null;

    const corners = [];
    let cx = 0, cy = 0;
    for (let j = 0; j < 4; j++) {
      const x = approx.data32S[j * 2], y = approx.data32S[j * 2 + 1];
      corners.push({ x, y });
      cx += x; cy += y;
    }
    quad = { corners, center: { x: cx / 4, y: cy / 4 }, area };
  } finally {
    approx.delete();
  }
  return quad;
}

// Stickers on one cube are similar in apparent size; drop outliers far from the
// median area, which are usually background clutter that happened to be square.
function filterByMedianArea(quads, [lo, hi]) {
  if (quads.length < 3) return quads;
  const areas = quads.map((q) => q.area).sort((a, b) => a - b);
  const median = areas[areas.length >> 1];
  return quads.filter((q) => q.area > median * lo && q.area < median * hi);
}

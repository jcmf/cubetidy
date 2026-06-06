// OpenCV-based cube detection for the auto-scan. BROWSER-ONLY in practice: real
// robustness can only be judged with a camera and a physical cube (`npm test`
// can't cover the pixel layer). The OpenCV `cv` module is passed in rather than
// imported, so this file has no side-effects and an API-level smoke test can feed
// it node's cv (see tools/detect-smoke.mjs).
//
// Stage 1 (here): find candidate sticker quadrilaterals in a frame — convex,
// roughly-square contours of plausible size. Two preprocessing methods:
//   'canny'  edge detection (good on high-contrast borders, weaker on cubes)
//   'mask'   HSV sticker mask: a sticker is SATURATED (the six colours) OR BRIGHT
//            (the white face), while the black grid/body is neither — so
//            (S > satThresh) OR (V > valThresh) isolates stickers from the grid and
//            from most backgrounds, and holds up for oblique faces. (Brightness
//            alone won't do: red and blue stickers are dark; white is desaturated.)
// Stage 2 (grouping the quads into the three faces of a corner-on view and solving
// a single 6-DoF cube pose) builds on this once stage 1 is dialled in on a cube.
//
// Every value here is overridable per-call (opts), so the ?detect harness can tune
// it live from the URL. All OpenCV Mats are explicitly deleted: this runs in the
// worker per frame and the Emscripten heap is not garbage-collected.

export const DETECT_DEFAULTS = {
  method: 'canny',      // 'canny' | 'mask'
  blur: 5,              // Gaussian blur kernel (odd); tames sensor noise

  // canny method
  cannyLo: 20,          // Canny hysteresis thresholds — tuned low on real cubes
  cannyHi: 50,          //   (neon/sticker tile borders are weak; hi=50 vs 90 went
                        //   from 1 detected face to 2 on the sample frames)
  dilateIters: 1,       // dilate edges to close gaps into loops (1: avoid merging
                        //   adjacent stickers into one blob)
  closeIters: 1,        // morphological close to bridge broken edges into closed
                        //   sticker loops (helps dim/oblique faces; 0 = off)

  // mask method (HSV; OpenCV ranges S,V in 0..255)
  satThresh: 60,        // S above this => a coloured sticker
  valThresh: 150,       // ...or V above this => a bright (white) sticker

  // shared quad gates
  approxEps: 0.08,      // approxPolyDP tolerance as a fraction of perimeter
  minAreaFrac: 0.0006,  // reject quads smaller than this fraction of the frame
  maxAreaFrac: 0.06,    // ...or larger (a whole face / background panel)
  minFill: 0.45,        // contourArea / min-area-rect area (lowered: sheared
                        //   oblique stickers don't fill their bounding rect)
  maxAspect: 3.0,       // max side ratio of the bounding rect (raised: foreshortened
                        //   far-face stickers are elongated)
  medianLo: 0.15,       // keep quads within [lo,hi]x the median area. Wide, because
  medianHi: 5.0,        //   corner-on perspective makes near/far stickers very
                        //   different sizes; a tight band culled the far face.
};

// Find candidate sticker quads in an ImageData. Returns an array of
// { corners: [{x,y}*4], center: {x,y}, area } in canvas pixel coordinates.
export function detectStickerQuads(cv, imageData, opts = {}) {
  const o = { ...DETECT_DEFAULTS, ...opts };
  const frameArea = imageData.width * imageData.height;

  const src = cv.matFromImageData(imageData);
  const work = new cv.Mat();    // gray (canny) or HSV (mask)
  const bin = new cv.Mat();     // binary image fed to findContours
  const tmp = new cv.Mat();     // scratch for the mask method
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const chans = new cv.MatVector(); // HSV planes for the mask method

  const quads = [];
  try {
    let retr;
    if (o.method === 'mask') {
      // sticker mask = (S > satThresh) OR (V > valThresh); the black grid is neither.
      cv.cvtColor(src, work, cv.COLOR_RGBA2RGB);
      cv.cvtColor(work, work, cv.COLOR_RGB2HSV);
      cv.split(work, chans);                       // H, S, V planes
      const S = chans.get(1), V = chans.get(2);
      cv.threshold(S, bin, o.satThresh, 255, cv.THRESH_BINARY);
      cv.threshold(V, tmp, o.valThresh, 255, cv.THRESH_BINARY);
      cv.bitwise_or(bin, tmp, bin);
      S.delete(); V.delete();
      cv.morphologyEx(bin, bin, cv.MORPH_OPEN, kernel); // clean speckle + thin bridges
      retr = cv.RETR_EXTERNAL;
    } else {
      cv.cvtColor(src, work, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(work, work, new cv.Size(o.blur | 1, o.blur | 1), 0);
      cv.Canny(work, bin, o.cannyLo, o.cannyHi);
      cv.dilate(bin, bin, kernel, new cv.Point(-1, -1), o.dilateIters);
      if (o.closeIters > 0) cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), o.closeIters);
      retr = cv.RETR_LIST;
    }

    cv.findContours(bin, contours, hierarchy, retr, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const quad = quadFromContour(cv, c, o, frameArea);
      if (quad) quads.push(quad);
      c.delete();
    }
  } finally {
    src.delete(); work.delete(); bin.delete(); tmp.delete();
    contours.delete(); hierarchy.delete(); kernel.delete(); chans.delete();
  }

  return filterByMedianArea(quads, o.medianLo, o.medianHi);
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
function filterByMedianArea(quads, lo, hi) {
  if (quads.length < 3) return quads;
  const areas = quads.map((q) => q.area).sort((a, b) => a - b);
  const median = areas[areas.length >> 1];
  return quads.filter((q) => q.area > median * lo && q.area < median * hi);
}

// Grid detection: compute the on-screen scan region and sample the 3x3 grid.
//
// MVP approach: a fixed centered square guide. The user aligns the cube face to
// it; we sample the average color of a small patch at each of the 9 cell centers.
// This module is the seam where real cube tracking (contour detection / pose
// estimation) would later plug in.

const REGION_FRACTION = 0.55; // square side as a fraction of the smaller dimension
const PATCH_FRACTION = 0.36;  // sampled patch size as a fraction of a cell

// Centered square region in canvas pixel coordinates.
export function computeRegion(width, height) {
  const side = Math.floor(Math.min(width, height) * REGION_FRACTION);
  return {
    x: Math.floor((width - side) / 2),
    y: Math.floor((height - side) / 2),
    side,
    cell: side / 3,
  };
}

// Average a small patch of pixels centered at (cx, cy).
function samplePatch(data, imgW, cx, cy, patch) {
  const half = Math.floor(patch / 2);
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = cy - half; y <= cy + half; y += 2) {
    for (let x = cx - half; x <= cx + half; x += 2) {
      const i = (y * imgW + x) * 4;
      r += data[i]; g += data[i + 1]; b += data[i + 2];
      n++;
    }
  }
  return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) };
}

// Sample 9 cells (row-major: top-left -> bottom-right) from the canvas context.
export function sampleGrid(ctx, region) {
  const { x, y, cell } = region;
  const img = ctx.getImageData(x, y, region.side, region.side);
  const patch = Math.max(4, Math.floor(cell * PATCH_FRACTION));
  const samples = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      // Cell center in region-local coordinates.
      const cx = Math.floor(cell * (col + 0.5));
      const cy = Math.floor(cell * (row + 0.5));
      samples.push(samplePatch(img.data, region.side, cx, cy, patch));
    }
  }
  return samples;
}

// --- corner-on scan --------------------------------------------------------
//
// The corner-on scan points a cube CORNER at the camera, so three faces are
// visible at once as foreshortened rhombi meeting at a central vertex. Two
// opposite-corner captures cover all six faces. Because the three faces are seen
// together, their relative orientation is fixed by geometry rather than by the
// user following per-face rotation instructions.
//
// A perfect corner-on cube projects to a regular hexagon silhouette. Its six
// outer vertices V0..V5 (V0 at top, going clockwise) plus the centre C are:
//
//        V0
//      /    \            V0 = top        V3 = bottom
//   V5        V1         V1 = upper-right V4 = lower-left
//    |   C    |          V2 = lower-right V5 = upper-left
//   V4        V2
//      \    /            The three visible faces are the rhombi
//        V3              (V5 V0 V1 C), (V5 C V3 V4), (C V1 V2 V3).
//
// Each rhombus is a parallelogram, so each face is sampled with a single affine
// map: a face is described by an origin corner O (where facelet 0 sits) and two
// edge vectors `col` (facelet col 0->2) and `row` (facelet row 0->2), each
// spanning the whole 3-cell edge. The corner->facelet labelling is derived so
// that the 9 samples come out in Kociemba facelet row-major order, letting the
// rest of the pipeline consume them unchanged. The derivation is guarded by
// test/corner-geometry.test.mjs.

const CORNER_FRACTION = 0.78; // hexagon width as a fraction of the smaller dim
const COS30 = Math.sqrt(3) / 2;

// The two corner captures and which face sits in each rhombus.
//   capture 1: URF corner toward camera — U on top, F lower-left, R lower-right
//   capture 2: opposite (DLB) corner    — D on top, L lower-left, B lower-right
// `o`/`c`/`r` name the hexagon points (centre 'C', outer 0..5) for the face's
// origin, col-edge end, and row-edge end. See the ASCII map above.
export const CORNER_CAPTURES = [
  {
    id: 'URF', title: 'first corner', faces: [
      { letter: 'U', o: 0, c: 1, r: 5 },
      { letter: 'F', o: 5, c: 'C', r: 4 },
      { letter: 'R', o: 'C', c: 1, r: 3 },
    ],
  },
  {
    id: 'DLB', title: 'opposite corner', faces: [
      { letter: 'D', o: 5, c: 0, r: 'C' },
      { letter: 'L', o: 3, c: 4, r: 'C' },
      { letter: 'B', o: 2, c: 3, r: 1 },
    ],
  },
];

// Hexagon centre + the seven named points for a given canvas size.
export function computeCornerRegion(width, height) {
  const r = Math.floor(Math.min(width, height) * CORNER_FRACTION) / 2;
  const cx = width / 2, cy = height / 2;
  const outer = [
    { x: cx, y: cy - r },                 // 0 top
    { x: cx + COS30 * r, y: cy - r / 2 }, // 1 upper-right
    { x: cx + COS30 * r, y: cy + r / 2 }, // 2 lower-right
    { x: cx, y: cy + r },                 // 3 bottom
    { x: cx - COS30 * r, y: cy + r / 2 }, // 4 lower-left
    { x: cx - COS30 * r, y: cy - r / 2 }, // 5 upper-left
  ];
  return { center: { x: cx, y: cy }, r, outer };
}

// The two captures are opposite corners; the user gets from the first to the
// second with ONE unambiguous motion: a 180° flip about the horizontal (left-
// right) screen axis (tip the top edge over until the opposite corner faces the
// camera). Derivation: the rotation taking the URF-corner pose to the DLB-corner
// pose works out to diag(1,-1,-1) in screen space. A 180° flip about horizontal
// has no directional ambiguity and keeps left-right fixed, so it fully pins the
// second pose (the user can only get it wrong by also twisting about the view
// axis).

// Resolve a point name ('C' or 0..5) to {x,y} for a region.
function point(region, name) {
  return name === 'C' ? region.center : region.outer[name];
}

// The affine frame for one face's rhombus: origin corner (facelet 0) and the two
// edge vectors spanning facelet col 0->2 and row 0->2.
export function faceFrame(region, face) {
  const o = point(region, face.o);
  const cEnd = point(region, face.c);
  const rEnd = point(region, face.r);
  return {
    o,
    col: { x: cEnd.x - o.x, y: cEnd.y - o.y },
    row: { x: rEnd.x - o.x, y: rEnd.y - o.y },
  };
}

// Map facelet (row i, col j) fractions to a screen point under a face frame.
function framePoint({ o, col, row }, fj, fi) {
  return { x: o.x + fj * col.x + fi * row.x, y: o.y + fj * col.y + fi * row.y };
}

// The 9 facelet sample points (row-major) for one face of a capture.
export function faceSamplePoints(region, face) {
  const frame = faceFrame(region, face);
  const pts = [];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      pts.push(framePoint(frame, (j + 0.5) / 3, (i + 0.5) / 3));
    }
  }
  return pts;
}

// Sample one corner capture, returning { [letter]: [9 samples] } in facelet order.
export function sampleCorner(ctx, region, capture) {
  const img = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const patch = Math.max(4, Math.floor(region.r * 0.12));
  const out = {};
  for (const face of capture.faces) {
    out[face.letter] = faceSamplePoints(region, face).map((p) =>
      samplePatch(img.data, ctx.canvas.width, Math.round(p.x), Math.round(p.y), patch));
  }
  return out;
}

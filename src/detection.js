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

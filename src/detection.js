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
// Geometry is a real perspective projection of a 3D cube, not a flat hexagon: a
// unit cube is rotated so a corner aims at the lens, then projected through a
// pinhole camera. The `persp` parameter (0..1) is the camera distance in cube-
// widths — 0 is (near-)orthographic, where the silhouette is a regular hexagon
// and cells are evenly spaced; higher values foreshorten (near corner magnified,
// far cells smaller), matching a cube held close to the lens. This is the seam a
// user-facing slider drives. Projecting the real sticker centres makes the 9
// samples per face come out in Kociemba facelet row-major order (per FACE_AXES),
// so the rest of the pipeline is unchanged. Guarded by corner-geometry.test.mjs.

const CORNER_FRACTION = 0.78; // silhouette width as a fraction of the smaller dim
const DZ_FAR = 16, DZ_NEAR = 3.2; // camera distance (cube half-size = 1) at persp 0 / 1

// Per-face in-plane axes matching Kociemba facelet orientation: outward normal
// `n`, plus the directions of increasing facelet column and row. A sticker
// centre is n + col*(2u-1) + row*(2v-1) for face-local (u,v) in [0,1].
const FACE_AXES = {
  U: { n: [0, 1, 0], col: [1, 0, 0], row: [0, 0, 1] },
  D: { n: [0, -1, 0], col: [1, 0, 0], row: [0, 0, -1] },
  F: { n: [0, 0, 1], col: [1, 0, 0], row: [0, -1, 0] },
  B: { n: [0, 0, -1], col: [-1, 0, 0], row: [0, -1, 0] },
  R: { n: [1, 0, 0], col: [0, 0, -1], row: [0, -1, 0] },
  L: { n: [-1, 0, 0], col: [0, 0, 1], row: [0, -1, 0] },
};

// Cube->camera rotations (camera frame: x right, y down, z toward viewer).
// Capture 0 aims the URF corner at the lens (U up, F lower-left, R lower-right).
// Capture 1 = capture 0 then a 180° flip about the horizontal screen axis
// (diag(1,-1,-1)), aiming the opposite DLB corner — ONE unambiguous motion (a
// 180° flip has no directional ambiguity and keeps left-right fixed).
const POSES = [
  [[0.70711, 0, -0.70711], [0.40825, -0.81650, 0.40825], [0.57735, 0.57735, 0.57735]],
  [[0.70711, 0, -0.70711], [-0.40825, 0.81650, -0.40825], [-0.57735, -0.57735, -0.57735]],
];

export const CORNER_CAPTURES = [
  { id: 'URF', title: 'first corner', faces: ['U', 'F', 'R'] },
  { id: 'DLB', title: 'opposite corner', faces: ['D', 'L', 'B'] },
];

// The 6 "equatorial" cube corners (all but the near + far corner of a corner-on
// view); their projection is the silhouette hexagon.
const EQUATORIAL = [[1, 1, -1], [-1, 1, 1], [1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]];

const clamp01 = (t) => Math.max(0, Math.min(1, t));

function matVec(m, p) {
  return [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
  ];
}

// Face-local (u,v) in [0,1]^2 -> a point on that cube face (side 2).
function facePoint(letter, u, v) {
  const { n, col, row } = FACE_AXES[letter];
  const su = 2 * u - 1, sv = 2 * v - 1;
  return [n[0] + col[0] * su + row[0] * sv, n[1] + col[1] * su + row[1] * sv, n[2] + col[2] * su + row[2] * sv];
}

// Build the projected scene (screen-space geometry) for one capture. Returns the
// centre, and per visible face its 9 cell-centre sample points (facelet row-
// major), the 4 rhombus corners, and the internal 3x3 grid lines for drawing.
export function computeCornerRegion(width, height, persp = 0, captureIndex = 0) {
  const Dz = DZ_FAR + (DZ_NEAR - DZ_FAR) * clamp01(persp);
  const pose = POSES[captureIndex];
  const cap = CORNER_CAPTURES[captureIndex];
  const raw = (p3) => { const c = matVec(pose, p3); const d = Dz - c[2]; return { x: c[0] / d, y: c[1] / d }; };

  // Fit the silhouette to the target radius, centred on the canvas.
  const outline = EQUATORIAL.map(raw);
  const radius = Math.max(...outline.map((p) => Math.hypot(p.x, p.y)));
  const scale = (Math.min(width, height) * CORNER_FRACTION / 2) / radius;
  const cx = width / 2, cy = height / 2;
  const map = (p3) => { const p = raw(p3); return { x: cx + p.x * scale, y: cy + p.y * scale }; };
  const at = (letter, u, v) => map(facePoint(letter, u, v));

  const faces = cap.faces.map((letter) => {
    const cells = [];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cells.push(at(letter, (j + 0.5) / 3, (i + 0.5) / 3));
    const quad = [at(letter, 0, 0), at(letter, 1, 0), at(letter, 1, 1), at(letter, 0, 1)];
    const grid = [];
    for (const k of [1, 2]) {
      grid.push([at(letter, k / 3, 0), at(letter, k / 3, 1)]);
      grid.push([at(letter, 0, k / 3), at(letter, 1, k / 3)]);
    }
    return { letter, cells, quad, grid };
  });

  return { center: { x: cx, y: cy }, scale, faces };
}

// Sample a capture's scene, returning { [letter]: [9 samples] } in facelet order.
// The patch size is derived per face from the on-screen cell spacing, so it
// tracks the foreshortening (cells shrink toward the far edges under perspective).
export function sampleCorner(ctx, scene) {
  const img = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
  const out = {};
  for (const face of scene.faces) {
    const cell = Math.hypot(face.cells[1].x - face.cells[0].x, face.cells[1].y - face.cells[0].y);
    const patch = Math.max(4, Math.floor(cell * 0.36));
    out[face.letter] = face.cells.map((p) =>
      samplePatch(img.data, ctx.canvas.width, Math.round(p.x), Math.round(p.y), patch));
  }
  return out;
}

// Offline overlay preview + visual diff check.
//
// Feeds the real overlay.js / glyph.js drawing code a skia-canvas 2D context and
// renders the canvas-drawn layer (scan guide, move arrows) plus the SVG glyphs to
// PNGs — no browser. This covers ONLY the overlay layer, not the composited app
// UI: there's no camera feed (the cube behind the overlays is synthetic), no CSS
// mirror, and no HTML chrome. It exists to catch overlay/glyph *geometry*
// regressions, which are easy to break and hard to spot by reading code.
//
//   npm run preview:update   render and overwrite the committed goldens
//   npm run preview:check     re-render and pixel-diff against the goldens,
//                             writing <scene>-diff.png for any mismatch (exit 1)
//
// Goldens live in tools/preview/golden/ (committed, so a geometry change shows up
// as a reviewable image diff). Fresh renders + diffs go to tools/preview/out/
// (gitignored). The diff is self-contained — it loads both PNGs with skia-canvas
// and compares pixels, so no extra dependency is needed.

import { Canvas, loadImage, FontLibrary } from 'skia-canvas';
import { mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { drawGuide, drawMove, drawClassified, drawCornerGuide } from '../src/overlay.js';
import { computeCornerRegion } from '../src/detection.js';
import { glyphSVG } from '../src/glyph.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const GOLDEN = join(HERE, 'preview', 'golden');
const OUT = join(HERE, 'preview', 'out');

// A pixel counts as changed if any channel moves more than this (absorbs
// sub-threshold antialiasing noise); a scene fails if more than BUDGET of its
// pixels change. Rendering is deterministic on a fixed platform, so a clean
// re-render diffs to zero; bump these if cross-platform AA jitter creeps in.
const CHANNEL_THRESHOLD = 8;
const BUDGET = 0.0005; // 0.05% of pixels

// A scrambled-looking face so each sticker shows a different color.
const FAKE_FACE = ['U', 'R', 'F', 'B', 'L', 'D', 'F', 'U', 'R'];

function region(x, y, side) {
  return { x, y, side, cell: side / 3 };
}

// Paint a black backing + synthetic cube face so overlays have something under them.
function fakeCube(ctx, reg) {
  ctx.save();
  ctx.fillStyle = '#000';
  ctx.fillRect(reg.x, reg.y, reg.side, reg.side);
  drawClassified(ctx, reg, FAKE_FACE);
  ctx.restore();
}

function label(ctx, text, cx, y) {
  ctx.save();
  ctx.fillStyle = '#e7ecf3';
  ctx.font = 'bold 24px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, cx, y);
  ctx.restore();
}

// --- scenes ----------------------------------------------------------------
// Each returns a finished Canvas. Add a scene here and it joins update + check.

async function sceneScanGuide() {
  const W = 720, H = 720, side = Math.floor(Math.min(W, H) * 0.55);
  const reg = region(Math.floor((W - side) / 2), Math.floor((H - side) / 2), side);
  const canvas = new Canvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, W, H);
  fakeCube(ctx, reg);
  drawGuide(ctx, reg, true);
  return canvas;
}

async function sceneMoveArrows() {
  const tokens = ['U', "U'", 'D', "D'", 'R', "R'", 'L', "L'", 'F', "F'", 'B', "B'"];
  const cols = 4, rows = 3, side = 200, gapX = 24, labelH = 44, gapY = 16, margin = 24;
  const W = margin * 2 + cols * side + (cols - 1) * gapX;
  const H = margin * 2 + rows * (side + labelH) + (rows - 1) * gapY;
  const canvas = new Canvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, W, H);
  tokens.forEach((tok, i) => {
    const c = i % cols, r = Math.floor(i / cols);
    const x = margin + c * (side + gapX);
    const y = margin + r * (side + labelH + gapY);
    const reg = region(x, y, side);
    fakeCube(ctx, reg);
    drawMove(ctx, reg, tok);
    label(ctx, tok, x + side / 2, y + side + 30);
  });
  return canvas;
}

async function sceneGlyphs() {
  const motions = ['spin', 'tiltTop', 'tiltBottom'];
  const box = 200, gap = 24, margin = 24, labelH = 44;
  const W = margin * 2 + motions.length * box + (motions.length - 1) * gap;
  const H = margin * 2 + box + labelH;
  const canvas = new Canvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < motions.length; i++) {
    const x = margin + i * (box + gap), y = margin;
    // The real glyph panel: translucent dark rounded rect (matches #glyph CSS).
    ctx.save();
    ctx.fillStyle = 'rgba(14,17,22,0.55)';
    ctx.strokeStyle = '#2c3444';
    ctx.beginPath();
    ctx.roundRect(x, y, box, box, 16);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    const svg = glyphSVG(motions[i]).replace('<svg ', '<svg width="120" height="120" ');
    const img = await loadImage(Buffer.from(svg, 'utf8'));
    const pad = 12;
    ctx.drawImage(img, x + pad, y + pad, box - 2 * pad, box - 2 * pad);
    label(ctx, motions[i], x + box / 2, y + box + 30);
  }
  return canvas;
}

async function sceneCornerGuide() {
  const W = 720, H = 720;
  const canvas = new Canvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, W, H);
  drawCornerGuide(ctx, computeCornerRegion(W, H), true);
  return canvas;
}

const SCENES = [
  { name: '1-scan-guide', render: sceneScanGuide },
  { name: '2-move-arrows', render: sceneMoveArrows },
  { name: '3-glyphs', render: sceneGlyphs },
  { name: '4-corner-guide', render: sceneCornerGuide },
];

// --- render + diff ---------------------------------------------------------

async function renderAll(dir) {
  mkdirSync(dir, { recursive: true });
  for (const s of SCENES) {
    const canvas = await s.render();
    await canvas.toFile(join(dir, `${s.name}.png`));
  }
}

// Decode a PNG back to raw RGBA via skia-canvas (no extra dependency).
async function pixels(path) {
  const img = await loadImage(path);
  const canvas = new Canvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return { w: img.width, h: img.height, data: ctx.getImageData(0, 0, img.width, img.height).data };
}

async function diffScene(name) {
  const goldenPath = join(GOLDEN, `${name}.png`);
  if (!existsSync(goldenPath)) return { status: 'missing' };

  const a = await pixels(goldenPath);
  const b = await pixels(join(OUT, `${name}.png`));
  if (a.w !== b.w || a.h !== b.h) {
    return { status: 'size', detail: `golden ${a.w}x${a.h} vs new ${b.w}x${b.h}` };
  }

  const diff = new Canvas(a.w, a.h);
  const dctx = diff.getContext('2d');
  const out = dctx.createImageData(a.w, a.h);
  let changed = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const moved = Math.abs(a.data[i] - b.data[i]) > CHANNEL_THRESHOLD ||
      Math.abs(a.data[i + 1] - b.data[i + 1]) > CHANNEL_THRESHOLD ||
      Math.abs(a.data[i + 2] - b.data[i + 2]) > CHANNEL_THRESHOLD;
    if (moved) {
      changed++;
      out.data[i] = 255; out.data[i + 1] = 0; out.data[i + 2] = 0; out.data[i + 3] = 255;
    } else {
      // Dimmed grayscale of the golden, for context behind the red.
      const g = (a.data[i] * 0.3 + a.data[i + 1] * 0.59 + a.data[i + 2] * 0.11) * 0.35;
      out.data[i] = out.data[i + 1] = out.data[i + 2] = g; out.data[i + 3] = 255;
    }
  }
  const pct = changed / (a.w * a.h);
  if (pct > BUDGET) {
    dctx.putImageData(out, 0, 0);
    await diff.toFile(join(OUT, `${name}-diff.png`));
    return { status: 'diff', changed, pct };
  }
  return { status: 'ok', changed, pct };
}

// --- main ------------------------------------------------------------------

const update = process.argv.includes('--update');
const font = FontLibrary.has('DejaVu Sans') ? 'DejaVu Sans' : (FontLibrary.families[0] ?? 'none');
console.log(`label font: ${font}`);

if (update) {
  await renderAll(GOLDEN);
  console.log(`\nupdated ${SCENES.length} golden(s) in tools/preview/golden/`);
  console.log('review the change with: git diff --stat -- tools/preview/golden');
} else {
  await renderAll(OUT);
  let failed = false;
  for (const s of SCENES) {
    const r = await diffScene(s.name);
    if (r.status === 'ok') {
      console.log(`  ok   ${s.name}${r.changed ? ` (${r.changed} px within budget)` : ''}`);
    } else if (r.status === 'diff') {
      console.log(`  DIFF ${s.name} — ${r.changed} px (${(r.pct * 100).toFixed(3)}%) → tools/preview/out/${s.name}-diff.png`);
      failed = true;
    } else if (r.status === 'size') {
      console.log(`  SIZE ${s.name} — ${r.detail}`);
      failed = true;
    } else {
      console.log(`  MISS ${s.name} — no golden; run: npm run preview:update`);
      failed = true;
    }
  }
  if (failed) {
    console.log('\nOverlay preview changed. If intentional: npm run preview:update (then review the golden diff).');
    process.exit(1);
  }
  console.log('\nall scenes match goldens');
}

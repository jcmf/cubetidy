// Verifies the corner-on scan geometry: the projected sample points from
// detection.js (computeCornerRegion) really describe three faces meeting at one
// cube corner, in correct Kociemba facelet order, with no transpose/flip/face-
// swap — and that this holds across the perspective range driven by the slider.
//
// Ground truth is the standard Kociemba corner/edge facelet adjacency (the
// physical gluing of the cube), hardcoded here independently of the geometry
// under test. The checks:
//   1. bijection   — the two captures' 6 faces x 9 cells tile all 54 facelets once
//   2. corner      — each capture's three centre-most stickers form a real corner
//   3. edge gluing — along each shared rhombus edge, the two faces' adjacent
//                    stickers line up as physically-glued pairs
// What this canNOT check is that the *physical holding instruction* matches (e.g.
// that "F lower-left" is really F) — that needs a real cube in the browser.
//
// Run: npm test

import { computeCornerRegion, CORNER_CAPTURES } from '../src/detection.js';

// Global facelet index = base[face] + local(0..8), in URFDLB order.
const BASE = { U: 0, R: 9, F: 18, D: 27, L: 36, B: 45 };
const g = (letter, local) => BASE[letter] + local;

// The 8 corner pieces and 12 edge pieces, as the facelets that meet on each.
// (Standard Kociemba adjacency — independent ground truth.)
const CORNERS = [
  ['U', 8, 'F', 2, 'R', 0], ['U', 6, 'F', 0, 'L', 2],
  ['U', 2, 'R', 2, 'B', 0], ['U', 0, 'L', 0, 'B', 2],
  ['D', 2, 'F', 8, 'R', 6], ['D', 0, 'F', 6, 'L', 8],
  ['D', 8, 'R', 8, 'B', 6], ['D', 6, 'L', 6, 'B', 8],
].map((t) => new Set([g(t[0], t[1]), g(t[2], t[3]), g(t[4], t[5])]));
const EDGES = [
  ['U', 7, 'F', 1], ['U', 5, 'R', 1], ['U', 1, 'B', 1], ['U', 3, 'L', 1],
  ['D', 1, 'F', 7], ['D', 5, 'R', 7], ['D', 7, 'B', 7], ['D', 3, 'L', 7],
  ['F', 5, 'R', 3], ['F', 3, 'L', 5], ['B', 3, 'R', 5], ['B', 5, 'L', 3],
].map((e) => new Set([g(e[0], e[1]), g(e[2], e[3])]));

// Do two global facelets belong to the same physical piece?
function glued(a, b) {
  return CORNERS.some((s) => s.has(a) && s.has(b)) ||
    EDGES.some((s) => s.has(a) && s.has(b));
}

let pass = 0, fail = 0;
function check(name, cond) {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name);
}

const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);

// Run the full check at a given perspective strength.
function runAt(persp) {
  const tag = `persp=${persp}`;
  const scenes = CORNER_CAPTURES.map((_, i) => computeCornerRegion(720, 720, persp, i));

  // 1. Bijection: all 54 facelets covered exactly once.
  const all = [];
  scenes.forEach((scene, si) =>
    scene.faces.forEach((f) => { for (let i = 0; i < 9; i++) all.push(g(f.letter, i)); }));
  check(`${tag}: two captures tile all 54 facelets exactly once`,
    all.length === 54 && new Set(all).size === 54);

  scenes.forEach((scene, si) => {
    const C = scene.center;
    const id = CORNER_CAPTURES[si].id;
    const faces = scene.faces.map((f) => ({ letter: f.letter, cells: f.cells, quad: f.quad }));

    // 2. The centre-most sticker of each face; the three must form a real corner.
    const centreGids = faces.map((f) => {
      let best = 0, bestD = Infinity;
      f.cells.forEach((p, i) => { const d = dist(p, C); if (d < bestD) { bestD = d; best = i; } });
      return g(f.letter, best);
    });
    const corner = CORNERS.find((s) => centreGids.every((x) => s.has(x)));
    check(`${tag} ${id}: three faces meet at one cube corner`, !!corner && new Set(centreGids).size === 3);

    // 3. Each face-pair shares a rhombus edge (centre -> a shared outer corner);
    //    the three sticker pairs along it must each be physically glued.
    for (let a = 0; a < faces.length; a++) {
      for (let b = a + 1; b < faces.length; b++) {
        const fa = faces[a], fb = faces[b];
        // Shared spoke end = the non-centre quad corner common to both faces.
        let shared = null, sharedD = Infinity;
        for (const pa of fa.quad) for (const pb of fb.quad) {
          if (dist(pa, C) < 1 || dist(pb, C) < 1) continue; // skip the centre corner
          const d = dist(pa, pb);
          if (d < sharedD) { sharedD = d; shared = pa; }
        }
        const dir = { x: shared.x - C.x, y: shared.y - C.y };
        const len = Math.hypot(dir.x, dir.y);
        const t = (p) => ((p.x - C.x) * dir.x + (p.y - C.y) * dir.y) / (len * len);
        const perp = (p) => Math.abs((p.x - C.x) * dir.y - (p.y - C.y) * dir.x) / len;
        const edgeRow = (f) => f.cells
          .map((p, i) => ({ i, t: t(p), d: perp(p) }))
          .sort((u, w) => u.d - w.d).slice(0, 3)
          .sort((u, w) => u.t - w.t).map((u) => g(f.letter, u.i));
        const ra = edgeRow(fa), rb = edgeRow(fb);
        const ok = [0, 1, 2].every((k) => glued(ra[k], rb[k]));
        check(`${tag} ${id}: ${fa.letter}-${fb.letter} edge stickers glue correctly`, ok);
      }
    }
  });
}

for (const persp of [0, 0.6]) runAt(persp);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

// Balanced, center-anchored classification (classifyFaces) vs plain nearest-
// center. Builds a cube whose red face has a few stickers pushed most of the way
// toward orange — the warm-light failure mode. Nearest-center mislabels them
// (counts come out wrong); the balanced k-means, knowing there are exactly nine
// of each color, forces them back to red.
//
// Run: npm test

import { buildReferences, classify, classifyFaces } from '../src/colors.js';

const CENTER = {
  U: { r: 245, g: 245, b: 245 }, D: { r: 255, g: 213, b: 0 },
  F: { r: 0, g: 155, b: 72 }, B: { r: 0, g: 70, b: 173 },
  R: { r: 183, g: 18, b: 52 }, L: { r: 255, g: 88, b: 0 },
};
const LETTERS = ['U', 'R', 'F', 'D', 'L', 'B'];
const lerp = (a, b, t) => ({
  r: Math.round(a.r + (b.r - a.r) * t),
  g: Math.round(a.g + (b.g - a.g) * t),
  b: Math.round(a.b + (b.b - a.b) * t),
});

let pass = 0, fail = 0;
function check(name, cond) {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name);
}

// Solved-cube faces: every face is nine copies of its center color.
function solvedFaces() {
  const f = {};
  for (const L of LETTERS) f[L] = Array.from({ length: 9 }, () => ({ ...CENTER[L] }));
  return f;
}

// Clean case: homogeneous colors classify back to themselves, no ambiguity.
{
  const { labels, counts, conflicts } = classifyFaces(solvedFaces());
  const allOwn = LETTERS.every((L) => labels[L].every((x) => x === L));
  check('clean cube: every sticker keeps its own color', allOwn);
  check('clean cube: nine of each color', LETTERS.every((L) => counts[L] === 9));
  check('clean cube: zero conflicts', conflicts === 0);
}

// Warm-light case: three non-center red stickers dragged 65% toward orange.
{
  const faces = solvedFaces();
  const ambiguous = lerp(CENTER.R, CENTER.L, 0.65);
  for (const idx of [0, 1, 2]) faces.R[idx] = { ...ambiguous };

  // Confirm the scenario is real: nearest-center sends those stickers to orange.
  const refs = buildReferences(faces);
  check('scenario: an ambiguous sticker reads as orange under nearest-center',
    classify(ambiguous, refs).label === 'L');
  const nearestCounts = {};
  for (const L of LETTERS) for (const s of faces[L]) {
    const lab = classify(s, refs).label;
    nearestCounts[lab] = (nearestCounts[lab] ?? 0) + 1;
  }
  check('nearest-center mislabels (red count != 9)', nearestCounts.R !== 9);

  // Balanced k-means recovers the true red face.
  const { labels, counts, conflicts } = classifyFaces(faces);
  check('balanced: red face fully recovered', labels.R.every((x) => x === 'R'));
  check('balanced: nine of every color', LETTERS.every((L) => counts[L] === 9));
  check('balanced: flags the ambiguity (conflicts > 0)', conflicts > 0);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

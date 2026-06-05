// End-to-end test of the scan -> classify -> facelet -> solve pipeline,
// exercising the real src modules. The camera/pixel-sampling layer is the only
// part not covered here (it needs a real browser + webcam).
//
// Run: npm test

import Cube from 'cubejs';
import { toFaceletString, validate, aggregateFaces } from '../src/cube-state.js';
import { solve } from '../src/solver.js';

// Distinct synthetic RGB per face letter so nearest-reference classify()
// round-trips each sticker back to its own letter.
const RGB = {
  U: { r: 245, g: 245, b: 245 }, D: { r: 255, g: 213, b: 0 },
  F: { r: 0, g: 155, b: 72 },    B: { r: 0, g: 70, b: 173 },
  R: { r: 183, g: 18, b: 52 },   L: { r: 255, g: 88, b: 0 },
};
const FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']; // facelet-string order

// Split a 54-char URFDLB string into the per-face 9-sample structure that the
// app builds while scanning.
function facesFromString(s) {
  const faces = {};
  let i = 0;
  for (const L of FACE_ORDER) {
    faces[L] = [];
    for (let k = 0; k < 9; k++) faces[L].push(RGB[s[i++]]);
  }
  return faces;
}

let pass = 0, fail = 0;
function check(name, cond) {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name);
}

Cube.initSolver();

for (let t = 0; t < 8; t++) {
  const scramble = new Cube();
  scramble.randomize();
  const original = scramble.asString();

  const { facelets, counts } = toFaceletString(facesFromString(original));
  check(`#${t} classify round-trips facelets`, facelets === original);
  check(`#${t} validate ok`, validate(facelets, counts).ok);

  const moves = await solve(facelets);
  const c = Cube.fromString(original);
  for (const m of moves) c.move(m);
  check(`#${t} solution (${moves.length} moves) solves the cube`, c.isSolved());
}

// Multi-pass aggregation: averaging passes with opposite per-channel noise must
// cancel back to the true colors and round-trip — the basis for re-scanning from
// new angles to ride out lighting (point 2).
{
  const scramble = new Cube();
  scramble.randomize();
  const truth = scramble.asString();
  const base = facesFromString(truth);
  const jitter = (faces, d) => {
    const out = {};
    for (const L of FACE_ORDER) {
      out[L] = faces[L].map((s) => ({ r: s.r + d, g: s.g - d, b: s.b + d }));
    }
    return out;
  };
  // Two noisy passes that bracket the truth, plus the clean truth as a third.
  const agg = aggregateFaces([jitter(base, 12), jitter(base, -12), base]);
  const { facelets, counts } = toFaceletString(agg);
  check('aggregated passes cancel noise and round-trip', facelets === truth);
  check('aggregated passes validate ok', validate(facelets, counts).ok);
  check('single-pass aggregate is identity',
    JSON.stringify(aggregateFaces([base])) === JSON.stringify(base));
}

// A misread that yields 10 of one color must be rejected before solving.
const broken = 'U'.repeat(10) + 'RRRRRRRRR' + 'FFFFFFFF' +
  'DDDDDDDDD' + 'LLLLLLLLL' + 'BBBBBBBBB';
check('broken color counts rejected',
  !validate(broken, { U: 10, R: 9, F: 8, D: 9, L: 9, B: 9 }).ok);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

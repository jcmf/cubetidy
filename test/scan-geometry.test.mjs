// Verifies the scan-orientation contract: if the user reorients the cube exactly
// as SCAN_STEPS instructs, the captured 3x3 grids reconstruct the true cube.
//
// Ground truth comes from cubejs whole-cube rotations (independent of the app's
// hand-derived geometry). For each step we rotate a scrambled cube as the step
// describes, read the face now facing the camera (the rotated cube's F facelets,
// in reading order), and feed those captures through the real assembly. The
// result must equal the original, unrotated state — otherwise a face is being
// stored transposed/rotated and solves would be silently wrong.
//
// Run: npm test

import Cube from 'cubejs';
import { SCAN_STEPS, toFaceletString, validate } from '../src/cube-state.js';

// Whole-cube rotation that brings each face to the front, matching the physical
// instruction in SCAN_STEPS (spin right->front for the sides; tilt for U/D).
const ROTATION = {
  F: [],                  // reference front, no rotation
  R: ['y'],               // spin: right face to front
  B: ['y', 'y'],          // spin again: back face to front
  L: ['y', 'y', 'y'],     // spin again: left face to front
  U: ["x'"],              // tilt top toward camera: up face to front
  D: ['x'],               // tilt bottom toward camera: down face to front
};

const RGB = {
  U: { r: 245, g: 245, b: 245 }, D: { r: 255, g: 213, b: 0 },
  F: { r: 0, g: 155, b: 72 },    B: { r: 0, g: 70, b: 173 },
  R: { r: 183, g: 18, b: 52 },   L: { r: 255, g: 88, b: 0 },
};

// The face the camera sees after a rotation = the rotated cube's F facelets
// (indices 18..26), in reading order — converted to synthetic samples.
function capture(state, rotation) {
  const c = Cube.fromString(state);
  for (const m of rotation) c.move(m);
  return [...c.asString().slice(18, 27)].map((ch) => RGB[ch]);
}

let pass = 0, fail = 0;
function check(name, cond) {
  cond ? pass++ : fail++;
  console.log((cond ? '  ok  ' : 'FAIL  ') + name);
}

// Guard: the test's rotation map must stay aligned with SCAN_STEPS.
check('SCAN_STEPS order matches rotation map',
  SCAN_STEPS.map((s) => s.face).join('') === 'FRBLUD' &&
  SCAN_STEPS.every((s) => ROTATION[s.face]));

Cube.initSolver();

for (let t = 0; t < 8; t++) {
  const cube = new Cube();
  cube.randomize();
  const truth = cube.asString();

  // Simulate scanning each step in order, storing under its face letter.
  const faces = {};
  for (const step of SCAN_STEPS) {
    faces[step.face] = capture(truth, ROTATION[step.face]);
  }

  const { facelets, counts } = toFaceletString(faces);
  check(`#${t} reconstructs the true cube from guided scans`, facelets === truth);
  check(`#${t} validate ok`, validate(facelets, counts).ok);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);

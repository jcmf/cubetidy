// Solver wrapper around cubejs (Kociemba two-phase algorithm).

import Cube from 'cubejs';

let initPromise = null;

// Build the pruning tables once. This is CPU-heavy (~1-2s); we run it off the
// initial paint and cache the promise so callers can await readiness.
export function initSolver() {
  if (!initPromise) {
    initPromise = new Promise((resolve) => {
      // Defer so the first frame can paint before we block the main thread.
      setTimeout(() => {
        Cube.initSolver();
        resolve();
      }, 0);
    });
  }
  return initPromise;
}

// Solve a 54-char URFDLB facelet string. Returns an array of move tokens,
// e.g. ['R', "U'", 'F2']. Throws if the cube state is invalid/unsolvable.
export async function solve(facelets) {
  await initSolver();
  const cube = Cube.fromString(facelets); // throws on malformed strings
  const solution = cube.solve();          // '' when already solved
  return solution.length ? solution.trim().split(/\s+/) : [];
}

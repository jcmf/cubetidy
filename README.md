# CubeTidy — AR Rubik's Cube Solver

A browser app that uses your camera to scan a Rubik's cube, computes the optimal
solution, and overlays arrows showing each move to make.

## Run

```bash
npm install
npm run dev
```

Open the printed `localhost` URL (camera access requires `https` or `localhost`).
To use it from a phone on your network, run `npm run dev -- --host` and open the
network URL **over https** (e.g. via a tunnel), since mobile browsers block the
camera on plain-http non-localhost origins.

## How to use

1. **Start camera** and grant permission.
2. **Scan all 6 faces.** Align each face to the on-screen grid and tap *Capture*.
   Follow the on-screen orientation hint for each face — the scan order is
   front → right → back → left → up → down, turning the cube *left* for the four
   sides, then tilting top/bottom toward the camera.
3. **Follow the arrows.** Hold the cube with the front center toward the camera
   and the top center up, then step through the moves with *Next* / *Prev*.
   Arrows show the layer and direction to turn.

## How it works

| Stage | Module | Notes |
|-------|--------|-------|
| Camera | `src/camera.js` | `getUserMedia`, prefers the rear camera |
| Sampling | `src/detection.js` | Fixed centered grid; averages a patch per cell |
| Color ID | `src/colors.js` | Calibrated nearest-reference match in CIE-Lab |
| State | `src/cube-state.js` | Assembles the Kociemba facelet string + validates |
| Solve | `src/solver.js` | [`cubejs`](https://github.com/ldez/cubejs) two-phase (Kociemba) |
| Overlay | `src/overlay.js` | Draws the guide grid and AR move arrows |

### Scope & next steps

This is the reliable "guided scan → solve → on-face arrows" design. The detection
module is the seam for upgrading toward continuous AR (contour detection + 6-DoF
pose via `solvePnP`, 3D arrows with three.js) so arrows can stick to a freely
moving cube. Color classification could also move to k-means clustering of all 54
samples to better separate red/orange under poor lighting.

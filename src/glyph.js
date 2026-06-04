// Isometric-cube instruction glyph for each scan step, as inline SVG.
//
// Rendered in an un-mirrored HTML overlay (not on the canvas), so the depicted
// rotation reads as a physical "rotate the cube this way" instruction regardless
// of the live preview's mirror state. Sits in a corner, clear of the grid.
//
// Iso cube (viewBox 120x120), three faces meeting at the near corner M(60,58):
//   top    rhombus  T(60,24)  UR(89.4,41) M       UL(30.6,41)
//   right  rhombus  UR        LR(89.4,75) Bt(60,92) M
//   left   rhombus  UL        M           Bt        LL(30.6,75)
const CUBE = `
  <polygon points="60,24 89.4,41 60,58 30.6,41" fill="#dfe5f0"/>
  <polygon points="89.4,41 89.4,75 60,92 60,58" fill="#97a4bd"/>
  <polygon points="30.6,41 30.6,75 60,92 60,58" fill="#6e7b95"/>
  <g fill="none" stroke="#222937" stroke-width="2" stroke-linejoin="round" stroke-linecap="round">
    <polygon points="60,24 89.4,41 89.4,75 60,92 30.6,75 30.6,41"/>
    <path d="M60,58 L30.6,41 M60,58 L89.4,41 M60,58 L60,92"/>
  </g>`;

// Amber stroke over a dark halo for contrast on any background.
function amber(d) {
  return `<path d="${d}" fill="none" stroke="#10141b" stroke-width="8" stroke-linecap="round"/>
          <path d="${d}" fill="none" stroke="#ffce1f" stroke-width="4.5" stroke-linecap="round" marker-end="url(#gh)"/>`;
}

// Spin about the vertical axis: a turntable ellipse below the cube, far half
// dashed, near half a solid arrow.
const SPIN = `
  <path d="M20,96 A40,9 0 0 1 100,96" fill="none" stroke="#c9d2e0" stroke-width="3"
        stroke-dasharray="5 5" opacity="0.65"/>
  ${amber('M100,96 A40,9 0 0 1 20,96')}`;

// Tilt about a horizontal axis: a curved arrow down (top toward camera) or up
// (bottom toward camera) over the front of the cube.
const ARROWS = {
  spin: SPIN,
  tiltTop: amber('M60,40 Q86,63 60,86'),
  tiltBottom: amber('M60,86 Q86,63 60,40'),
};

export function glyphSVG(motion) {
  if (!ARROWS[motion]) return '';
  return `<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="gh" markerUnits="userSpaceOnUse" markerWidth="14" markerHeight="14"
              refX="11" refY="7" orient="auto">
        <path d="M1,1 L13,7 L1,13 Z" fill="#ffce1f"/>
      </marker>
    </defs>
    ${CUBE}
    ${ARROWS[motion]}
  </svg>`;
}

// Edge mode preprocessing: landmarks -> the 128x128 skeleton the CNN takes, in the browser.
//
// This is a port of gestureflow/edge.py, which is the reference. It reproduces numpy's
// float32 arithmetic (Math.fround) and round-half-to-even, so the rendered pixels are
// identical to the Python reference -- `pnpm test:edge` checks every sample in the dataset.
// The renderer differs slightly from OpenCV's line drawing on the server; how much that
// matters is measured in docs/edge_parity.json. No DOM or browser APIs, so Node can test it.

export const CANVAS = 128; // gestureflow/core/preprocessing.py CANVAS
export const MARGIN = 0.12; // ... MARGIN
const BONE_RADIUS = 1.5; // gestureflow/edge.py EDGE_BONE_RADIUS
const JOINT_RADIUS = 3; // ... EDGE_JOINT_RADIUS

export const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// BGR, like OpenCV, because that is the channel order the model was trained on.
function jointColorBGR(i: number): [number, number, number] {
  if (i === 0) return [200, 200, 200];
  if (i <= 4) return [80, 80, 255];
  if (i <= 8) return [80, 220, 255];
  if (i <= 12) return [80, 255, 120];
  if (i <= 16) return [255, 200, 80];
  return [255, 110, 200];
}

const f32 = Math.fround;

/** numpy.round: halves go to the nearest even integer. */
function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** preprocessing.normalize in float32: wrist to the origin, furthest joint at distance 1. */
export function normalize(landmarks: number[][]): Float32Array {
  const pts = new Float32Array(42);
  const wx = f32(landmarks[0][0]);
  const wy = f32(landmarks[0][1]);
  let scale = 0;
  for (let i = 0; i < 21; i++) {
    const x = f32(f32(landmarks[i][0]) - wx);
    const y = f32(f32(landmarks[i][1]) - wy);
    pts[2 * i] = x;
    pts[2 * i + 1] = y;
    const len = f32(Math.sqrt(f32(f32(x * x) + f32(y * y))));
    if (len > scale) scale = len;
  }
  if (scale < 1e-6) scale = f32(1e-6);
  for (let i = 0; i < 42; i++) pts[i] = f32(pts[i] / scale);
  return pts;
}

/** Normalized points -> integer canvas coordinates, exactly as preprocessing.render. */
export function pixelPoints(pts: Float32Array): Int32Array {
  const half = CANVAS / 2;
  const s = f32(half * (1 - MARGIN));
  const out = new Int32Array(42);
  for (let i = 0; i < 42; i++) out[i] = roundHalfEven(f32(f32(pts[i] * s) + half));
  return out;
}

/** The skeleton as an HxWx3 BGR byte image, same layout as the numpy array. */
export function render(pts: Float32Array): Uint8Array {
  const c = CANVAS;
  const img = new Uint8Array(c * c * 3);
  const xy = pixelPoints(pts);
  const r2 = BONE_RADIUS * BONE_RADIUS;
  const pad = Math.ceil(BONE_RADIUS);
  const paint = (x: number, y: number, col: [number, number, number]) => {
    const o = (y * c + x) * 3;
    img[o] = col[0];
    img[o + 1] = col[1];
    img[o + 2] = col[2];
  };

  for (const [a, b] of CONNECTIONS) {
    const x0 = xy[2 * a], y0 = xy[2 * a + 1], x1 = xy[2 * b], y1 = xy[2 * b + 1];
    const dx = x1 - x0, dy = y1 - y0;
    const length2 = dx * dx + dy * dy;
    const xa = Math.max(0, Math.min(x0, x1) - pad), xb = Math.min(c - 1, Math.max(x0, x1) + pad);
    const ya = Math.max(0, Math.min(y0, y1) - pad), yb = Math.min(c - 1, Math.max(y0, y1) + pad);
    const col = jointColorBGR(b);
    for (let py = ya; py <= yb; py++) {
      for (let px = xa; px <= xb; px++) {
        const t = length2 > 0 ? Math.min(1, Math.max(0, ((px - x0) * dx + (py - y0) * dy) / length2)) : 0;
        const ex = px - (x0 + t * dx);
        const ey = py - (y0 + t * dy);
        if (ex * ex + ey * ey <= r2) paint(px, py, col);
      }
    }
  }
  const jr = JOINT_RADIUS;
  for (let i = 0; i < 21; i++) {
    const x = xy[2 * i], y = xy[2 * i + 1];
    const col = jointColorBGR(i);
    for (let py = Math.max(0, y - jr); py <= Math.min(c - 1, y + jr); py++) {
      for (let px = Math.max(0, x - jr); px <= Math.min(c - 1, x + jr); px++) {
        if ((px - x) ** 2 + (py - y) ** 2 <= jr * jr) paint(px, py, col);
      }
    }
  }
  return img;
}

/** HxWx3 bytes -> 3xHxW float32 in 0..1, the layout the CNN takes (preprocessing.to_input). */
export function toInput(img: Uint8Array): Float32Array {
  const n = CANVAS * CANVAS;
  const out = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) {
    for (let ch = 0; ch < 3; ch++) out[ch * n + p] = img[p * 3 + ch] / 255;
  }
  return out;
}

export function softmax(logits: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

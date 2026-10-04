// In-browser hand tracking with MediaPipe. Camera frames stay on this device; only the 21
// landmarks are sent to the server.

import type { HandLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

// Served from /public/mediapipe by scripts/setup-mediapipe.mjs, so nothing loads from a CDN.
const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/mediapipe/hand_landmarker.task";

let landmarker: Promise<HandLandmarker> | null = null;

export function loadHandLandmarker(): Promise<HandLandmarker> {
  landmarker ??= (async () => {
    const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    return HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
  })().catch((e) => {
    landmarker = null; // allow a retry
    throw e;
  });
  return landmarker;
}

/** MediaPipe's normalized landmarks -> pixel coordinates of the unmirrored frame, which is
 *  the coordinate space the model was trained in. */
export function toPixels(points: NormalizedLandmark[], width: number, height: number): number[][] {
  return points.map((p) => [p.x * width, p.y * height]);
}

// Same 21 bones and finger colours as gestureflow/core/preprocessing.py.
export const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export function jointColor(i: number): string {
  if (i === 0) return "rgb(200,200,200)"; // wrist
  if (i <= 4) return "rgb(255,80,80)"; // thumb
  if (i <= 8) return "rgb(255,220,80)"; // index
  if (i <= 12) return "rgb(120,255,80)"; // middle
  if (i <= 16) return "rgb(80,200,255)"; // ring
  return "rgb(200,110,255)"; // pinky
}

/** Draw the skeleton over the video. `mirror` matches a CSS-mirrored preview. */
export function drawSkeleton(ctx: CanvasRenderingContext2D, points: NormalizedLandmark[] | null, mirror: boolean) {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  if (!points) return;
  const xy = points.map((p) => [(mirror ? 1 - p.x : p.x) * width, p.y * height]);
  ctx.lineWidth = Math.max(2, width / 240);
  for (const [a, b] of CONNECTIONS) {
    ctx.strokeStyle = jointColor(b);
    ctx.beginPath();
    ctx.moveTo(xy[a][0], xy[a][1]);
    ctx.lineTo(xy[b][0], xy[b][1]);
    ctx.stroke();
  }
  const r = Math.max(3, width / 160);
  xy.forEach(([x, y], i) => {
    ctx.fillStyle = jointColor(i);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  });
}

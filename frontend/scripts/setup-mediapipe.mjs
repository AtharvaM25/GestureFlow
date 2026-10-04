// Puts the files the browser needs for on-device ML into public/, so the app serves them
// itself instead of loading them from a CDN. Runs on `pnpm install` (postinstall); re-run
// any time with `node scripts/setup-mediapipe.mjs`.
//
//   public/mediapipe/  MediaPipe hand tracking: WASM runtime + hand_landmarker.task
//   public/ort/        ONNX Runtime Web: WASM runtime                       (Edge mode)
//   public/edge/       the CNN exported to ONNX + its labels, from ../models (Edge mode)

import { copyFileSync, cpSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pub = join(root, "public");
const modules = join(root, "node_modules");
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// --- MediaPipe
mkdirSync(join(pub, "mediapipe"), { recursive: true });
cpSync(join(modules, "@mediapipe", "tasks-vision", "wasm"), join(pub, "mediapipe", "wasm"), { recursive: true });
const landmarker = join(pub, "mediapipe", "hand_landmarker.task");
if (!existsSync(landmarker)) {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`mediapipe: model download failed (${res.status})`);
  writeFileSync(landmarker, Buffer.from(await res.arrayBuffer()));
}
console.log("setup: mediapipe ready");

// --- ONNX Runtime Web (the plain-WASM build needs just these two files)
mkdirSync(join(pub, "ort"), { recursive: true });
for (const f of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(modules, "onnxruntime-web", "dist", f), join(pub, "ort", f));
}
console.log("setup: onnxruntime-web ready");

// --- the exported model (python -m gestureflow.export_onnx writes it to ../models)
const models = process.env.GESTUREFLOW_MODELS_DIR ?? join(root, "..", "models");
mkdirSync(join(pub, "edge"), { recursive: true });
for (const f of ["gesture_cnn.onnx", "gesture_cnn.json"]) {
  if (existsSync(join(models, f))) copyFileSync(join(models, f), join(pub, "edge", f));
  else console.warn(`setup: ${f} not found in ${models}; Edge mode will be unavailable`);
}
console.log("setup: edge model ready");

// Checks the browser's Edge-mode code against the Python reference:
//   - render(): identical pixels to gestureflow/edge.py for every sample in data/landmarks.csv
//   - LetterCommitter: identical output to gestureflow/core/temporal.py on recorded streams
//   - the whole pipeline with ONNX Runtime Web: same probabilities as Python (golden samples)
// Run: pnpm test:edge   (regenerate the fixture with `python -m gestureflow.edge`)

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { CANVAS, normalize, render, softmax, toInput } from "../src/lib/edge.ts";
import { LetterCommitter } from "../src/lib/temporal.ts";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const fixture = JSON.parse(readFileSync(join(root, "frontend", "src", "lib", "edge-fixture.json"), "utf8"));

let failures = 0;
const fail = (msg: string) => {
  failures++;
  if (failures <= 10) console.error("FAIL", msg);
};

// --- renderer, every sample
const [header, ...rows] = readFileSync(join(root, fixture.csv), "utf8").trim().split(/\r?\n/);
const cols = header.split(",");
const xIdx = Array.from({ length: 21 }, (_, i) => cols.indexOf(`x${i}`));
const yIdx = Array.from({ length: 21 }, (_, i) => cols.indexOf(`y${i}`));
if (rows.length !== fixture.render_sha256.length) fail(`csv has ${rows.length} rows, fixture ${fixture.render_sha256.length}`);
rows.forEach((row, r) => {
  const v = row.split(",");
  const landmarks = xIdx.map((xi, i) => [Number(v[xi]), Number(v[yIdx[i]])]);
  const hash = createHash("sha256").update(render(normalize(landmarks))).digest("hex");
  if (hash !== fixture.render_sha256[r]) fail(`render differs from Python on row ${r}`);
});
console.log(`render: ${rows.length} samples checked`);

// --- letter commits
let frames = 0;
let commits = 0;
fixture.temporal.forEach((s: { frames: ([string, number] | null)[]; expected: [string | null, string | null, number][] }, k: number) => {
  const c = new LetterCommitter();
  s.frames.forEach((f, i) => {
    let got: [string | null, string | null, number];
    if (f === null) {
      c.noHand();
      got = [null, null, 0];
    } else {
      const step = c.update(f[0], f[1]);
      got = [step.smoothed, step.committed, c.progress];
    }
    const want = s.expected[i];
    if (got[0] !== want[0] || got[1] !== want[1] || Math.abs(got[2] - want[2]) > 1e-12) {
      fail(`stream ${k} frame ${i}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
    }
    if (want[1]) commits++;
    frames++;
  });
});
console.log(`temporal: ${frames} frames, ${commits} commits checked`);

// --- whole pipeline: raw landmarks -> TS preprocessing -> ONNX Runtime Web -> probabilities
const ort = await import("onnxruntime-web/wasm");
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(readFileSync(join(root, "models", "gesture_cnn.onnx")));
let worst = 0;
for (const c of fixture.pipeline as { label: string; raw: number[][]; probs: number[] }[]) {
  const input = new ort.Tensor("float32", toInput(render(normalize(c.raw))), [1, 3, CANVAS, CANVAS]);
  const probs = softmax((await session.run({ skeleton: input })).logits.data as Float32Array);
  probs.forEach((p, i) => (worst = Math.max(worst, Math.abs(p - c.probs[i]))));
}
if (worst > 1e-4) fail(`pipeline probabilities differ from Python by up to ${worst}`);
console.log(`pipeline: ${fixture.pipeline.length} golden samples, max probability difference ${worst.toExponential(1)}`);

if (failures) {
  console.error(`${failures} mismatch(es)`);
  process.exit(1);
}
console.log("edge: all checks passed");

// Edge mode inference: the ONNX export of the CNN, run in the browser by ONNX Runtime Web.
// The model and runtime are served by this app (scripts/setup-mediapipe.mjs copies them).

import type { InferenceSession } from "onnxruntime-web";

import { CANVAS, MARGIN, normalize, render, softmax, toInput } from "@/lib/edge";

type Meta = { labels: string[]; preprocessing: { canvas: number; margin: number; align_rotation: boolean } };

export type EdgePrediction = {
  gesture: string;
  confidence: number;
  alternatives: { gesture: string; confidence: number }[];
  latency_ms: number;
};

export class EdgeModel {
  private constructor(
    private session: InferenceSession,
    private ort: typeof import("onnxruntime-web"),
    readonly labels: string[],
  ) {}

  static async load(): Promise<EdgeModel> {
    const ort = await import("onnxruntime-web/wasm");
    ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation headers; one is plenty here
    const meta: Meta = await fetch("/edge/gesture_cnn.json").then((r) => r.json());
    // Same guard as the server: refuse a model exported under different preprocessing.
    const p = meta.preprocessing;
    if (p.canvas !== CANVAS || p.margin !== MARGIN || p.align_rotation) {
      throw new Error("Edge model was exported with different preprocessing settings.");
    }
    const session = await ort.InferenceSession.create("/edge/gesture_cnn.onnx", { executionProviders: ["wasm"] });
    return new EdgeModel(session, ort as unknown as typeof import("onnxruntime-web"), meta.labels);
  }

  async predict(landmarks: number[][]): Promise<EdgePrediction> {
    const t0 = performance.now();
    const input = new this.ort.Tensor("float32", toInput(render(normalize(landmarks))), [1, 3, CANVAS, CANVAS]);
    const out = await this.session.run({ skeleton: input });
    const probs = softmax(out.logits.data as Float32Array);
    const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
    return {
      gesture: this.labels[order[0]],
      confidence: probs[order[0]],
      alternatives: order.slice(0, 3).map((i) => ({ gesture: this.labels[i], confidence: probs[i] })),
      latency_ms: performance.now() - t0,
    };
  }
}

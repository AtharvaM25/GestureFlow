import { ArrowRight, Cpu, Lock, Radio, ScanLine } from "lucide-react";
import Link from "next/link";

import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const STEPS = [
  { icon: ScanLine, title: "Track", body: "MediaPipe finds 21 hand landmarks in your browser." },
  { icon: Radio, title: "Stream", body: "Only those 42 numbers go to the server, over a WebSocket." },
  { icon: Cpu, title: "Classify", body: "Landmarks are normalized, redrawn as a skeleton and classified by a CNN." },
  { icon: Lock, title: "Commit", body: "A letter is added after 15 steady, confident frames, then saved to your session." },
];

// Every number here is printed by a script in the repository: docs/evaluation_report.json,
// docs/calibration_report.json and the latency benchmark in docs/API.md.
const METRICS = [
  { value: "94.2%", label: "accuracy on a recording session the model never saw" },
  { value: "97.3%", label: "after a 5-frame calibration to your hand" },
  { value: "29 ms", label: "median server time per frame (CPU)" },
  { value: "0", label: "camera frames sent anywhere. Edge mode sends nothing at all" },
];

export default function Landing() {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-20 px-4 py-20">
      <section className="flex max-w-3xl flex-col gap-6">
        <span className="w-fit rounded-full border border-border px-3 py-1 text-xs text-muted-foreground">
          Hand-sign letters, recognized in real time
        </span>
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          Sign a word. Watch it appear.
          <span className="block text-muted-foreground">Your camera never leaves your device.</span>
        </h1>
        <p className="max-w-2xl text-lg text-muted-foreground">
          GestureFlow reads hand-sign letters from your webcam. Your browser tracks the hand and sends only its 21
          landmark points, so the server never sees a single frame of video.
        </p>
        <div className="flex gap-3">
          <Link href="/try" className={buttonVariants({ size: "lg" })}>
            Try it now, no account <ArrowRight />
          </Link>
          <Link href="/register" className={buttonVariants({ size: "lg", variant: "outline" })}>
            Create an account
          </Link>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {METRICS.map((m) => (
          <Card key={m.label}>
            <CardContent className="flex flex-col gap-1">
              <span className="font-mono text-3xl font-semibold">{m.value}</span>
              <span className="text-sm text-muted-foreground">{m.label}</span>
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="flex flex-col gap-6">
        <h2 className="text-2xl font-semibold tracking-tight">How it works</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map((s, i) => (
            <Card key={s.title}>
              <CardContent className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <s.icon className="size-4" /> Step {i + 1}
                </div>
                <h3 className="font-medium">{s.title}</h3>
                <p className="text-sm text-muted-foreground">{s.body}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}

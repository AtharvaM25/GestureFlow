"use client";

import { Camera, CameraOff, Delete, ShieldCheck, Sparkles, Volume2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api, ApiError, tokenStore, wsUrl, type Alternative, type Session } from "@/lib/api";
import { EdgeModel } from "@/lib/edge-model";
import { drawSkeleton, loadHandLandmarker, toPixels } from "@/lib/hand";
import { LetterCommitter } from "@/lib/temporal";
import { cn } from "@/lib/utils";

type Mode = "server" | "edge";
type Phase = "idle" | "starting" | "live" | "stopped";

type Result = {
  hand: boolean;
  gesture: string | null;
  confidence: number | null;
  stability: number;
  committed: string | null;
  alternatives: Alternative[];
  latency_ms: number; // server time (server mode) or in-browser model time (edge mode)
  text: string;
  calibrated?: boolean; // server mode: the user's calibration was applied
};

const CLOSE_REASONS: Record<number, string> = {
  4401: "The server rejected your sign-in. Sign in again.",
  4404: "This session has ended or does not exist.",
  4400: "The server rejected a message.",
};

const PRIVACY: Record<Mode, string> = {
  server: "Server mode: only hand landmarks leave this device, never camera frames.",
  edge: "Edge mode: recognition runs in your browser. Nothing is sent anywhere and nothing is saved.",
};

function speak(text: string) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

/** The live recognition screen. `guest`: no account -- Edge mode only, nothing saved. */
export function Recognizer({ guest = false }: { guest?: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const edgeRef = useRef<EdgeModel | null>(null);
  const committerRef = useRef(new LetterCommitter());
  const edgeTextRef = useRef("");
  const inFlight = useRef(false); // one frame at a time, so a slow server or model never builds a backlog
  const lastVideoTime = useRef(-1);
  const fpsWindow = useRef<number[]>([]);
  const sentAt = useRef(0);

  const [mode, setMode] = useState<Mode>(guest ? "edge" : "server");
  const [sentencesOn, setSentencesOn] = useState(true);
  const [phase, setPhase] = useState<Phase>("idle");
  const [session, setSession] = useState<Session | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [text, setText] = useState("");
  const [history, setHistory] = useState<{ letter: string; confidence: number; at: number }[]>([]);
  const [rtt, setRtt] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [flash, setFlash] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sentence, setSentence] = useState<string | null>(null);
  const [sentenceBusy, setSentenceBusy] = useState(false);

  const teardown = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    inFlight.current = false;
  }, []);

  useEffect(() => teardown, [teardown]);

  // hide "Make a sentence" when the server has sentence generation turned off
  useEffect(() => {
    if (guest) return;
    api.health().then((h) => setSentencesOn(h.sentence_provider !== "none")).catch(() => {});
  }, [guest]);

  // Shared by both modes: show a frame's result.
  const apply = useCallback((r: Result) => {
    inFlight.current = false;
    const now = performance.now();
    setRtt(now - sentAt.current);
    const w = fpsWindow.current;
    w.push(now);
    while (w.length && now - w[0] > 1000) w.shift();
    setFps(w.length);
    setResult(r);
    setText(r.text);
    if (r.committed) {
      const letter = r.committed;
      setHistory((h) => [{ letter, confidence: r.confidence ?? 0, at: Date.now() }, ...h].slice(0, 30));
      setFlash(true);
      setTimeout(() => setFlash(false), 600);
    }
  }, []);

  // Edge mode: the same steps as the server, run in the browser.
  const runEdge = useCallback(
    async (landmarks: number[][] | null) => {
      const committer = committerRef.current;
      if (!landmarks) {
        committer.noHand();
        apply({ hand: false, gesture: null, confidence: null, stability: 0, committed: null, alternatives: [], latency_ms: 0, text: edgeTextRef.current });
        return;
      }
      const p = await edgeRef.current!.predict(landmarks);
      const step = committer.update(p.gesture, p.confidence);
      if (step.committed) edgeTextRef.current += step.committed;
      apply({ hand: true, gesture: step.smoothed, confidence: p.confidence, stability: committer.progress, committed: step.committed, alternatives: p.alternatives, latency_ms: p.latency_ms, text: edgeTextRef.current });
    },
    [apply],
  );

  const loop = useCallback(
    async (m: Mode) => {
      const landmarker = await loadHandLandmarker();
      const tick = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !streamRef.current) return;
        if (video.readyState >= 2 && video.currentTime !== lastVideoTime.current) {
          lastVideoTime.current = video.currentTime;
          const points = landmarker.detectForVideo(video, performance.now()).landmarks[0] ?? null;
          if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          drawSkeleton(canvas.getContext("2d")!, points, true);
          // pixel coordinates of the unmirrored frame -- the space the model was trained in
          const landmarks = points ? toPixels(points, video.videoWidth, video.videoHeight) : null;
          if (!inFlight.current) {
            const ws = wsRef.current;
            if (m === "edge") {
              inFlight.current = true;
              sentAt.current = performance.now();
              runEdge(landmarks).catch((e) => {
                inFlight.current = false;
                setError(e instanceof Error ? e.message : "Edge inference failed.");
              });
            } else if (ws?.readyState === WebSocket.OPEN) {
              inFlight.current = true;
              sentAt.current = performance.now();
              ws.send(JSON.stringify({ type: "frame", landmarks, client_ts: sentAt.current }));
            }
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    [runEdge],
  );

  const onMessage = useCallback(
    (event: MessageEvent) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "result") apply(msg);
      else if (msg.type === "text") {
        setText(msg.text);
        setHistory((h) => h.slice(1));
      } else if (msg.type === "error") {
        inFlight.current = false;
        setError(msg.message);
      }
    },
    [apply],
  );

  async function start() {
    setError(null);
    setSentence(null);
    setHistory([]);
    setText("");
    setResult(null);
    setSession(null);
    committerRef.current = new LetterCommitter();
    edgeTextRef.current = "";
    setPhase("starting");
    try {
      if (mode === "server") await api.me(); // fails fast with "sign in again" if the token has expired
      // camera and models first, so a denied camera doesn't leave an empty session behind
      const [stream] = await Promise.all([
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false }),
        loadHandLandmarker(),
        mode === "edge" && !edgeRef.current ? EdgeModel.load().then((m) => (edgeRef.current = m)) : null,
      ]);
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      if (mode === "edge") {
        setPhase("live");
        loop("edge");
        return;
      }

      const s = await api.createSession();
      setSession(s);
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => ws.send(JSON.stringify({ type: "auth", token: tokenStore.get(), session_id: s.id }));
      ws.onmessage = (e) => {
        if (JSON.parse(e.data).type === "ready") {
          ws.onmessage = onMessage;
          setPhase("live");
          loop("server");
        }
      };
      ws.onclose = (e) => {
        if (e.code !== 1000 && e.code !== 1005) {
          setError(CLOSE_REASONS[e.code] ?? "Lost connection to the recognition server.");
          teardown();
          setPhase("stopped");
        }
      };
    } catch (e) {
      teardown();
      setPhase("idle");
      if (e instanceof DOMException && e.name === "NotAllowedError") setError("Camera access was denied.");
      else if (e instanceof DOMException && e.name === "NotFoundError") setError("No camera was found.");
      else setError(e instanceof Error ? e.message : "Could not start recognition.");
    }
  }

  async function stop() {
    teardown();
    setPhase("stopped");
    canvasRef.current?.getContext("2d")?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    if (session) await api.endSession(session.id).catch(() => {});
  }

  function backspace() {
    if (mode === "edge") {
      edgeTextRef.current = edgeTextRef.current.slice(0, -1);
      setText(edgeTextRef.current);
      setHistory((h) => h.slice(1));
    } else {
      wsRef.current?.send(JSON.stringify({ type: "backspace" }));
    }
  }

  async function makeSentence() {
    if (!session) return;
    setSentenceBusy(true);
    try {
      setSentence((await api.sentence(session.id)).sentence);
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 503
          ? `Sentence generation is unavailable right now (${e.message}).`
          : e instanceof Error
            ? e.message
            : "Sentence generation failed.",
      );
    } finally {
      setSentenceBusy(false);
    }
  }

  const live = phase === "live";

  return (
    <>
      <PageHeader
        title={guest ? "Try GestureFlow" : "Recognize"}
        description={
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck className="size-4 text-emerald-400" /> {PRIVACY[mode]}
          </span>
        }
        actions={
          <div className="flex items-center gap-2">
            <div className="flex rounded-lg border border-border p-0.5 text-xs" role="group" aria-label="Inference mode">
              {(guest ? (["edge"] as const) : (["server", "edge"] as const)).map((m) => (
                <button
                  key={m}
                  type="button"
                  disabled={phase === "live" || phase === "starting"}
                  onClick={() => setMode(m)}
                  aria-pressed={mode === m}
                  className={cn(
                    "rounded-md px-2.5 py-1 capitalize transition-colors disabled:cursor-not-allowed",
                    mode === m ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {m}
                </button>
              ))}
            </div>
            {live ? (
              <Button variant="outline" onClick={stop}>
                <CameraOff /> Stop
              </Button>
            ) : (
              <Button onClick={start} disabled={phase === "starting"}>
                <Camera /> {phase === "starting" ? "Starting…" : phase === "stopped" ? "New session" : "Start camera"}
              </Button>
            )}
          </div>
        }
      />

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="flex flex-col gap-4">
          <div
            className={cn(
              "relative aspect-[4/3] overflow-hidden rounded-xl border bg-black transition-colors",
              flash ? "border-emerald-400" : "border-border",
            )}
          >
            {/* the preview is mirrored like a selfie; the landmarks used for recognition are not */}
            <video ref={videoRef} playsInline muted className="absolute inset-0 size-full -scale-x-100 object-cover" />
            <canvas ref={canvasRef} className="absolute inset-0 size-full object-cover" />
            {!live && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
                <Camera className="size-8" />
                {phase === "starting" ? "Loading hand tracking…" : "Camera is off"}
              </div>
            )}
            {live && result && (
              <div className="absolute left-3 top-3 flex gap-2">
                <Badge variant={result.hand ? "default" : "secondary"}>{result.hand ? "Hand" : "No hand"}</Badge>
                <Badge variant="secondary" className="font-mono">{fps} fps</Badge>
                <Badge variant="outline" className="capitalize">{mode}</Badge>
                {result.calibrated && <Badge variant="outline">Calibrated</Badge>}
              </div>
            )}
          </div>

          <Card>
            <CardContent className="flex items-center gap-3">
              <div className="min-h-9 flex-1 font-mono text-2xl tracking-widest">
                {text || <span className="text-base tracking-normal text-muted-foreground">Hold a sign steady to add a letter…</span>}
              </div>
              <Button variant="outline" size="icon" aria-label="Speak the text" disabled={!text} onClick={() => speak(text)}>
                <Volume2 />
              </Button>
              <Button variant="outline" size="icon" aria-label="Delete last letter" disabled={!live || !text} onClick={backspace}>
                <Delete />
              </Button>
            </CardContent>
          </Card>

          {phase === "stopped" && session && sentencesOn && (
            <Card>
              <CardContent className="flex flex-wrap items-center gap-3">
                <Button variant="secondary" onClick={makeSentence} disabled={!text || sentenceBusy}>
                  <Sparkles /> {sentenceBusy ? "Writing…" : "Make a sentence"}
                </Button>
                {sentence && (
                  <>
                    <p className="flex-1">{sentence}</p>
                    <Button variant="outline" size="icon" aria-label="Speak the sentence" onClick={() => speak(sentence)}>
                      <Volume2 />
                    </Button>
                  </>
                )}
                <Link href={`/sessions/${session.id}`} className="ml-auto text-sm text-muted-foreground hover:text-foreground">
                  View session →
                </Link>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Prediction</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex items-end justify-between">
                <span className="font-mono text-6xl font-semibold leading-none">{result?.gesture ?? "–"}</span>
                <span className="font-mono text-sm text-muted-foreground">
                  {result?.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : ""}
                </span>
              </div>
              <div className="flex flex-col gap-1.5">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Hold progress</span>
                  <span className="font-mono">{Math.round((result?.stability ?? 0) * 100)}%</span>
                </div>
                <Progress value={(result?.stability ?? 0) * 100} />
              </div>
              <div className="flex flex-col gap-1.5">
                <span className="text-xs text-muted-foreground">Alternatives</span>
                {(result?.alternatives ?? []).map((a) => (
                  <div key={a.gesture} className="flex items-center gap-2 text-sm">
                    <span className="w-6 font-mono">{a.gesture}</span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                      <div className="h-full bg-primary" style={{ width: `${a.confidence * 100}%` }} />
                    </div>
                    <span className="w-12 text-right font-mono text-xs text-muted-foreground">
                      {(a.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
                {!result?.alternatives?.length && <span className="text-sm text-muted-foreground">—</span>}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Latency</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">{mode === "edge" ? "Model (in browser)" : "Server"}</span>
                <span className="font-mono">{result?.hand ? `${result.latency_ms.toFixed(1)} ms` : "–"}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground">{mode === "edge" ? "Frame total" : "Round trip"}</span>
                <span className="font-mono">{rtt !== null ? `${rtt.toFixed(1)} ms` : "–"}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>History</CardTitle>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <span className="text-sm text-muted-foreground">Committed letters appear here.</span>
              ) : (
                <ul className="flex flex-col gap-1 text-sm">
                  {history.map((h) => (
                    <li key={h.at} className="flex justify-between">
                      <span className="font-mono">{h.letter}</span>
                      <span className="font-mono text-xs text-muted-foreground">{(h.confidence * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}

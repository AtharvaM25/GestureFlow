"use client";

import { Camera, CameraOff, Check, RotateCcw, ShieldCheck, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { formatDate, PageHeader } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type CalibrationStatus } from "@/lib/api";
import { drawSkeleton, loadHandLandmarker, toPixels } from "@/lib/hand";
import { cn } from "@/lib/utils";

const FRAMES_PER_LETTER = 5; // what docs/calibration_report.json measured
const FRAME_GAP_MS = 120; // spread the captures over ~half a second of holding the sign

export default function CalibrationPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const latest = useRef<number[][] | null>(null); // most recent landmarks, pixel coords
  const capturing = useRef<{ letter: string; frames: number[][][]; last: number } | null>(null);

  const [status, setStatus] = useState<CalibrationStatus | null>(null);
  const [live, setLive] = useState(false);
  const [hand, setHand] = useState(false);
  const [samples, setSamples] = useState<Record<string, number[][][]>>({});
  const [current, setCurrent] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.calibration().then(setStatus).catch((e) => setError(e.message));
  }, []);

  const stopCamera = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setLive(false);
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const labels = status?.labels ?? [];
  const letter = labels[current];
  const done = labels.filter((l) => (samples[l]?.length ?? 0) >= FRAMES_PER_LETTER).length;

  async function startCamera() {
    setError(null);
    try {
      const [stream, landmarker] = await Promise.all([
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false }),
        loadHandLandmarker(),
      ]);
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setLive(true);
      let lastTime = -1;
      const tick = () => {
        const canvas = canvasRef.current;
        if (!streamRef.current || !canvas) return;
        if (video.readyState >= 2 && video.currentTime !== lastTime) {
          lastTime = video.currentTime;
          const points = landmarker.detectForVideo(video, performance.now()).landmarks[0] ?? null;
          if (canvas.width !== video.videoWidth) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }
          drawSkeleton(canvas.getContext("2d")!, points, true);
          latest.current = points ? toPixels(points, video.videoWidth, video.videoHeight) : null;
          setHand(points !== null);

          // while capturing, take a frame every FRAME_GAP_MS that has a hand in it
          const cap = capturing.current;
          const now = performance.now();
          if (cap && latest.current && now - cap.last >= FRAME_GAP_MS) {
            cap.frames.push(latest.current);
            cap.last = now;
            if (cap.frames.length >= FRAMES_PER_LETTER) {
              const { letter: l, frames } = cap;
              capturing.current = null;
              setSamples((s) => ({ ...s, [l]: frames }));
              setCurrent((i) => Math.min(i + 1, labels.length - 1));
              setBusy(null);
            }
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e) {
      stopCamera();
      setError(e instanceof DOMException && e.name === "NotAllowedError" ? "Camera access was denied." : e instanceof Error ? e.message : "Could not start the camera.");
    }
  }

  function capture() {
    if (!letter || capturing.current) return;
    capturing.current = { letter, frames: [], last: 0 };
    setBusy("capturing");
  }

  async function save() {
    setBusy("saving");
    setError(null);
    try {
      setStatus(await api.saveCalibration(samples));
      setSamples({});
      setCurrent(0);
      stopCamera();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Saving failed.");
    } finally {
      setBusy(null);
    }
  }

  async function remove() {
    await api.deleteCalibration().catch((e) => setError(e.message));
    setStatus(await api.calibration());
  }

  return (
    <>
      <PageHeader
        title="Calibration"
        description={
          <span className="inline-flex items-center gap-1.5">
            <ShieldCheck className="size-4 text-emerald-400" /> Sign each letter once so recognition adapts to your hand.
            Only an average of the model&apos;s features per letter is stored, not your landmarks.
          </span>
        }
      />

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {status && (
        <Card>
          <CardContent className="flex flex-wrap items-center gap-3 text-sm">
            {status.calibrated ? (
              <>
                <Badge>Calibrated</Badge>
                <span className="text-muted-foreground">
                  {status.samples_per_label} frames per letter, saved {status.created_at && formatDate(status.created_at)}. Applied in
                  Server mode.
                </span>
              </>
            ) : status.stale ? (
              <>
                <Badge variant="secondary">Out of date</Badge>
                <span className="text-muted-foreground">The model changed since you calibrated. Record it again.</span>
              </>
            ) : (
              <>
                <Badge variant="secondary">Not calibrated</Badge>
                <span className="text-muted-foreground">
                  On a recording session the model hadn&apos;t seen, 5 frames per letter raised accuracy from 92.7% to 97.3%.
                </span>
              </>
            )}
            {(status.calibrated || status.stale) && (
              <Button variant="outline" size="sm" className="ml-auto" onClick={remove}>
                <Trash2 /> Remove
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="relative aspect-[4/3] overflow-hidden rounded-xl border border-border bg-black">
          <video ref={videoRef} playsInline muted className="absolute inset-0 size-full -scale-x-100 object-cover" />
          <canvas ref={canvasRef} className="absolute inset-0 size-full object-cover" />
          {!live && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
              <Camera className="size-8" />
              <Button onClick={startCamera} disabled={!status}>
                <Camera /> Start camera
              </Button>
            </div>
          )}
          {live && (
            <div className="absolute left-3 top-3">
              <Badge variant={hand ? "default" : "secondary"}>{hand ? "Hand" : "No hand"}</Badge>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle>
                Letter {Math.min(current + 1, labels.length)} of {labels.length}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-4">
              <span className="font-mono text-7xl font-semibold">{letter ?? "–"}</span>
              <p className="text-center text-sm text-muted-foreground">
                Hold the sign for <span className="font-mono">{letter}</span>, then press Capture. It takes {FRAMES_PER_LETTER} frames
                over about half a second.
              </p>
              <div className="flex w-full gap-2">
                <Button className="flex-1" onClick={capture} disabled={!live || !hand || busy !== null}>
                  {busy === "capturing" ? "Capturing…" : "Capture"}
                </Button>
                <Button variant="outline" size="icon" aria-label="Previous letter" disabled={current === 0 || busy !== null}
                        onClick={() => setCurrent((i) => i - 1)}>
                  <RotateCcw />
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>
                Progress {done}/{labels.length}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="grid grid-cols-7 gap-1.5">
                {labels.map((l, i) => {
                  const ok = (samples[l]?.length ?? 0) >= FRAMES_PER_LETTER;
                  return (
                    <button
                      key={l}
                      type="button"
                      onClick={() => setCurrent(i)}
                      className={cn(
                        "flex h-8 items-center justify-center rounded-md border font-mono text-sm",
                        i === current ? "border-primary" : "border-border",
                        ok ? "bg-emerald-500/15 text-emerald-400" : "text-muted-foreground",
                      )}
                    >
                      {ok ? <Check className="size-3.5" /> : l}
                    </button>
                  );
                })}
              </div>
              <Button onClick={save} disabled={done < labels.length || busy !== null}>
                {busy === "saving" ? "Saving…" : "Save calibration"}
              </Button>
              {live && (
                <Button variant="ghost" onClick={stopCamera}>
                  <CameraOff /> Stop camera
                </Button>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}

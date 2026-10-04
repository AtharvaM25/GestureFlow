"use client";

import { useEffect, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Gesture } from "@/lib/api";
import { cn } from "@/lib/utils";

function f1Tone(f1: number) {
  if (f1 >= 0.95) return "text-emerald-400";
  if (f1 >= 0.8) return "text-amber-400";
  return "text-red-400";
}

export default function GesturesPage() {
  const [gestures, setGestures] = useState<Gesture[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.gestures().then(setGestures).catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <PageHeader
        title="Gestures"
        description="The letters the model recognizes. F1 is measured on a recording session the model never saw during training."
      />
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 lg:grid-cols-9">
        {!gestures
          ? Array.from({ length: 18 }, (_, i) => <Skeleton key={i} className="h-24" />)
          : gestures.map((g) => (
              <Card key={g.label} size="sm">
                <CardContent className="flex flex-col gap-2">
                  <span className="font-mono text-3xl font-semibold">{g.label}</span>
                  {g.f1_unseen_session !== null ? (
                    <span className={cn("font-mono text-sm", f1Tone(g.f1_unseen_session))}>
                      F1 {g.f1_unseen_session.toFixed(2)}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-foreground">not evaluated</span>
                  )}
                </CardContent>
              </Card>
            ))}
      </div>
    </>
  );
}

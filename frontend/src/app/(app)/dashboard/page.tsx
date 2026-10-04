"use client";

import { Play } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { DailyColumns, LetterBars } from "@/components/charts";
import { formatDate, PageHeader, Stat } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type Analytics, type Health, type Session } from "@/lib/api";
import { cn } from "@/lib/utils";

const RANGES = [7, 30, 90] as const;

export default function DashboardPage() {
  const [days, setDays] = useState<(typeof RANGES)[number]>(30);
  const [stats, setStats] = useState<Analytics | null>(null);
  const [loadingStats, setLoadingStats] = useState(true);
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.sessions().then(setSessions).catch((e) => setError(e.message));
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .analytics(days)
      .then((a) => !cancelled && setStats(a))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoadingStats(false));
    return () => {
      cancelled = true;
    };
  }, [days]);

  function pickRange(d: (typeof RANGES)[number]) {
    setLoadingStats(true); // keep the previous render, dimmed, while the new range loads
    setDays(d);
  }

  return (
    <>
      <PageHeader
        title="Dashboard"
        description={
          health ? (
            <span>
              Server {health.status === "ok" ? "online" : "degraded"} · {health.model_classes} letters · database {health.database}
            </span>
          ) : (
            "Your recognition activity."
          )
        }
        actions={
          <Link href="/recognize" className={buttonVariants()}>
            <Play /> New session
          </Link>
        }
      />

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* one filter row, scoping everything in this section */}
      <div className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Activity in the last</span>
        <div className="flex rounded-lg border border-border p-0.5 text-xs" role="group" aria-label="Time range">
          {RANGES.map((d) => (
            <button
              key={d}
              type="button"
              aria-pressed={days === d}
              onClick={() => pickRange(d)}
              className={cn("rounded-md px-2.5 py-1", days === d ? "bg-muted text-foreground" : "text-muted-foreground hover:text-foreground")}
            >
              {d} days
            </button>
          ))}
        </div>
      </div>

      <div className={cn("flex flex-col gap-6 transition-opacity", loadingStats && stats && "opacity-60")}>
        <div className="grid gap-4 sm:grid-cols-5">
          <Stat label="Sessions" value={stats ? stats.sessions : "–"} />
          <Stat label="Letters" value={stats ? stats.letters : "–"} />
          <Stat label="Frames" value={stats ? stats.frames.toLocaleString() : "–"} />
          <Stat label="Avg confidence" value={stats?.avg_confidence != null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "–"} hint="when a letter commits" />
          <Stat label="Avg server time" value={stats?.avg_latency_ms != null ? `${stats.avg_latency_ms.toFixed(1)} ms` : "–"} hint="when a letter commits" />
        </div>

        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <Card>
            <CardHeader>
              <CardTitle>Letters per day</CardTitle>
            </CardHeader>
            <CardContent>{stats ? <DailyColumns data={stats.per_day} /> : <Skeleton className="h-48" />}</CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Letters you sign most</CardTitle>
            </CardHeader>
            <CardContent>
              {!stats ? (
                <Skeleton className="h-48" />
              ) : stats.per_letter.length ? (
                <LetterBars data={stats.per_letter.slice(0, 10)} />
              ) : (
                <p className="py-8 text-center text-sm text-muted-foreground">No letters in this period.</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent sessions</CardTitle>
        </CardHeader>
        <CardContent>
          {!sessions ? (
            <div className="flex flex-col gap-2">
              <Skeleton className="h-8" />
              <Skeleton className="h-8" />
            </div>
          ) : sessions.length === 0 ? (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No sessions yet. <Link href="/recognize" className="text-foreground underline-offset-4 hover:underline">Start one</Link> to
              sign your first letters.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Started</TableHead>
                  <TableHead>Text</TableHead>
                  <TableHead className="text-right">Frames</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sessions.map((s) => (
                  <TableRow key={s.id}>
                    <TableCell>
                      <Link href={`/sessions/${s.id}`} className="hover:underline">
                        {formatDate(s.started_at)}
                      </Link>
                    </TableCell>
                    <TableCell className="font-mono">{s.text || <span className="text-muted-foreground">—</span>}</TableCell>
                    <TableCell className="text-right font-mono">{s.frame_count.toLocaleString()}</TableCell>
                    <TableCell>
                      <Badge variant={s.ended_at ? "secondary" : "default"}>{s.ended_at ? "Ended" : "Open"}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </>
  );
}

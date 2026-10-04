"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { formatDate, PageHeader, Stat } from "@/components/page-header";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type SessionDetail } from "@/lib/api";

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const [s, setS] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.session(Number(id)).then(setS).catch((e) => setError(e.message));
  }, [id]);

  if (error)
    return (
      <Alert variant="destructive">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  if (!s) return <Skeleton className="h-64" />;

  const avgConf = s.letters.length ? s.letters.reduce((n, l) => n + l.confidence, 0) / s.letters.length : null;
  const avgLat = s.letters.length ? s.letters.reduce((n, l) => n + l.latency_ms, 0) / s.letters.length : null;

  return (
    <>
      <PageHeader
        title={`Session #${s.id}`}
        description={`${formatDate(s.started_at)} · model ${s.model_version}`}
        actions={<Link href="/dashboard" className="text-sm text-muted-foreground hover:text-foreground">← Dashboard</Link>}
      />
      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Letters" value={s.letters.length} />
        <Stat label="Frames" value={s.frame_count.toLocaleString()} />
        <Stat label="Avg confidence" value={avgConf !== null ? `${(avgConf * 100).toFixed(1)}%` : "–"} hint="at the commit frame" />
        <Stat label="Avg server time" value={avgLat !== null ? `${avgLat.toFixed(1)} ms` : "–"} hint="at the commit frame" />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Text</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          <p className="font-mono text-2xl tracking-widest">{s.text || "—"}</p>
          {s.sentence && <p className="text-muted-foreground">{s.sentence}</p>}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Committed letters</CardTitle>
        </CardHeader>
        <CardContent>
          {s.letters.length === 0 ? (
            <p className="text-sm text-muted-foreground">No letters were committed in this session.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Letter</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Runner-up</TableHead>
                  <TableHead className="text-right">Server time</TableHead>
                  <TableHead>When</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {s.letters.map((l) => (
                  <TableRow key={l.id}>
                    <TableCell className="font-mono text-lg">{l.gesture}</TableCell>
                    <TableCell className="font-mono">{(l.confidence * 100).toFixed(1)}%</TableCell>
                    <TableCell className="font-mono text-muted-foreground">
                      {l.alternatives[1] ? `${l.alternatives[1].gesture} ${(l.alternatives[1].confidence * 100).toFixed(1)}%` : "—"}
                    </TableCell>
                    <TableCell className="text-right font-mono">{l.latency_ms.toFixed(1)} ms</TableCell>
                    <TableCell className="text-muted-foreground">{new Date(l.created_at).toLocaleTimeString()}</TableCell>
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

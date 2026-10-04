"use client";

// Two small single-series charts in plain HTML. One series each, so one hue (--viz-series-1)
// and no legend: the card title names the series. Bars are at most 24px thick with a 4px
// rounded data end and a square baseline; every mark has a hover/focus tooltip, and each
// chart has a table view so no value depends on hovering.

import { useState, type ReactNode } from "react";

import { cn } from "@/lib/utils";

type Tip = { x: number; y: number; value: string; label: string } | null;

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div
      role="status"
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs shadow-md"
      style={{ left: tip.x, top: tip.y - 6 }}
    >
      {/* value leads, label follows */}
      <div className="font-semibold text-foreground">{tip.value}</div>
      <div className="text-muted-foreground">{tip.label}</div>
    </div>
  );
}

function TableView({ head, rows }: { head: [string, string]; rows: [ReactNode, ReactNode][] }) {
  return (
    <details className="mt-3 text-sm">
      <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">View as table</summary>
      <table className="mt-2 w-full text-left">
        <thead className="text-xs text-muted-foreground">
          <tr>
            <th className="py-1 font-normal">{head[0]}</th>
            <th className="py-1 text-right font-normal">{head[1]}</th>
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {rows.map(([a, b], i) => (
            <tr key={i} className="border-t border-border">
              <td className="py-1">{a}</td>
              <td className="py-1 text-right">{b}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

const short = (iso: string) => new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

/** Letters committed per day: columns growing from one baseline. */
export function DailyColumns({ data }: { data: { date: string; letters: number }[] }) {
  const [tip, setTip] = useState<Tip>(null);
  const max = Math.max(1, ...data.map((d) => d.letters));
  const show = (e: { currentTarget: HTMLElement }, d: { date: string; letters: number }) => {
    const r = e.currentTarget.getBoundingClientRect();
    const box = e.currentTarget.parentElement!.getBoundingClientRect();
    const h = (d.letters / max) * box.height;
    setTip({ x: r.left - box.left + r.width / 2, y: box.height - h, value: `${d.letters} letter${d.letters === 1 ? "" : "s"}`, label: short(d.date) });
  };
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs text-muted-foreground">
        <span className="tabular-nums">{max}</span>
      </div>
      <div className="relative" onPointerLeave={() => setTip(null)}>
        {/* recessive gridline at the max, axis at the baseline */}
        <div className="absolute inset-x-0 top-0 border-t border-dashed" style={{ borderColor: "var(--viz-grid)" }} />
        <div className="relative flex h-40 items-end justify-between gap-[2px] border-b" style={{ borderColor: "var(--viz-axis)" }}>
          {data.map((d) => (
            <div
              key={d.date}
              tabIndex={0}
              aria-label={`${short(d.date)}: ${d.letters} letters`}
              className="group flex h-full max-w-6 flex-1 cursor-default items-end justify-center outline-none"
              onPointerEnter={(e) => show(e, d)}
              onFocus={(e) => show(e, d)}
              onBlur={() => setTip(null)}
            >
              <div
                className="w-full rounded-t-[4px] transition-opacity group-hover:opacity-80 group-focus-visible:opacity-80"
                style={{ height: `${(d.letters / max) * 100}%`, minHeight: d.letters ? 2 : 0, background: "var(--viz-series-1)" }}
              />
            </div>
          ))}
        </div>
        <Tooltip tip={tip} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground">
        <span>{data.length ? short(data[0].date) : ""}</span>
        <span>{data.length ? `${short(data[data.length - 1].date)} (UTC days)` : ""}</span>
      </div>
      <TableView head={["Day", "Letters"]} rows={data.map((d) => [short(d.date), d.letters])} />
    </div>
  );
}

/** Letters you sign most: horizontal bars, value at the tip. */
export function LetterBars({ data }: { data: { gesture: string; count: number; avg_confidence: number }[] }) {
  const [tip, setTip] = useState<Tip>(null);
  const max = Math.max(1, ...data.map((d) => d.count));
  const show = (e: { currentTarget: HTMLElement }, d: (typeof data)[number]) => {
    const r = e.currentTarget.getBoundingClientRect();
    const box = e.currentTarget.closest("[data-chart]")!.getBoundingClientRect();
    setTip({ x: r.left - box.left + 40, y: r.top - box.top, value: `${d.count} × ${d.gesture}`, label: `avg confidence ${(d.avg_confidence * 100).toFixed(1)}%` });
  };
  return (
    <div>
      <div data-chart className="relative flex flex-col gap-[2px]" onPointerLeave={() => setTip(null)}>
        {data.map((d) => (
          <div
            key={d.gesture}
            tabIndex={0}
            aria-label={`${d.gesture}: ${d.count}, average confidence ${(d.avg_confidence * 100).toFixed(1)}%`}
            className="group flex cursor-default items-center gap-3 rounded-sm py-0.5 outline-none focus-visible:bg-muted"
            onPointerEnter={(e) => show(e, d)}
            onFocus={(e) => show(e, d)}
            onBlur={() => setTip(null)}
          >
            <span className="w-5 text-center font-mono text-sm">{d.gesture}</span>
            <div className="flex flex-1 items-center gap-2 border-l" style={{ borderColor: "var(--viz-axis)" }}>
              <div
                className={cn("h-4 rounded-r-[4px] transition-opacity group-hover:opacity-80")}
                style={{ width: `${(d.count / max) * 100}%`, background: "var(--viz-series-1)" }}
              />
              <span className="text-xs tabular-nums text-muted-foreground">{d.count}</span>
            </div>
          </div>
        ))}
        <Tooltip tip={tip} />
      </div>
      <TableView
        head={["Letter", "Count · avg confidence"]}
        rows={data.map((d) => [d.gesture, `${d.count} · ${(d.avg_confidence * 100).toFixed(1)}%`])}
      />
    </div>
  );
}

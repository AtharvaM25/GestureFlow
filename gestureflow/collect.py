"""
Confusion-aware landmark collector.

Replaces collecter.py's flat `--target 150` per class with a recording plan derived from
what the model actually gets wrong. Three signals decide what you record next:

  1. cross-session F1   from docs/evaluation_report.json -- how badly the class fails on a
                        session the model has never seen. This is the honest signal.
  2. landmark spread    average distance of each sample from its class mean. High spread
                        means the gesture was held inconsistently during capture.
  3. session coverage   a class recorded in one sitting has no evidence of generalizing.

Also fixes a data-loss bug in collecter.py: rows accumulated in memory and were written
only after a clean `q` exit, so Ctrl-C, a crash or closing the window discarded the whole
sitting. Here every sample is appended to the CSV immediately and fsync'd.

    python -m gestureflow.collect --plan                 # what to record, no camera needed
    python -m gestureflow.collect --label K --session day3
    python -m gestureflow.collect --auto --session day3  # walk the priority queue
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from gestureflow.core import preprocessing
from gestureflow.paths import EVAL_REPORT, LANDMARKS_CSV

COORD_COLS = [f"{a}{i}" for i in range(21) for a in "xy"]
CSV_COLS = [f"{a}{i}" for i in range(21) for a in "xy"]
HEADER = ["label", "session", "timestamp", "seq"] + COORD_COLS

BURST_EVERY = 3
DEFAULT_TARGET = 150
MIN_SESSIONS = 3          # below this a class has no cross-session evidence
F1_URGENT = 0.80          # cross-session F1 under this is a priority class
SPREAD_HIGH = 0.12        # above this the gesture was held inconsistently


# ----------------------------------------------------------------- planning
@dataclass
class ClassNeed:
    label: str
    samples: int
    sessions: list[str]
    spread: float
    f1: float | None = None
    reasons: list[str] = field(default_factory=list)
    priority: float = 0.0

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)


def class_spread(coords: np.ndarray) -> float:
    """Mean distance of each normalized sample from the class mean. Same measure as
    gestureflow/preview.py, so the numbers are comparable."""
    if len(coords) == 0:
        return 0.0
    normed = np.stack([preprocessing.normalize(c) for c in coords])
    return float(np.linalg.norm(normed - normed.mean(axis=0), axis=2).mean())


def load_f1(report_path: Path) -> tuple[dict[str, float], str | None]:
    """Per-class F1 from an evaluation report, plus a note describing its provenance."""
    if not report_path.exists():
        return {}, None
    try:
        rep = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: could not read {report_path.name}: {e}", file=sys.stderr)
        return {}, None

    folds = rep.get("folds", {})
    if not folds:
        return {}, None

    # Average per-class F1 across folds; with one fold this is just that fold.
    acc: dict[str, list[float]] = {}
    for fold in folds.values():
        for label, f1 in fold.get("per_class_f1", {}).items():
            acc.setdefault(label, []).append(float(f1))
    merged = {k: sum(v) / len(v) for k, v in acc.items()}
    note = f"{rep.get('protocol', 'unknown protocol')}, {len(folds)} fold(s)"
    return merged, note


def build_plan(df: pd.DataFrame, f1: dict[str, float], target: int) -> list[ClassNeed]:
    needs: list[ClassNeed] = []

    for label, g in df.groupby("label"):
        coords = g[CSV_COLS].to_numpy(np.float32).reshape(-1, 21, 2)
        need = ClassNeed(
            label=str(label),
            samples=len(g),
            sessions=sorted(str(s) for s in g["session"].unique()),
            spread=class_spread(coords),
            f1=f1.get(str(label)),
        )

        score = 0.0
        if need.f1 is not None and need.f1 < F1_URGENT:
            # Dominant term: a class that fails on an unseen session is the whole problem.
            score += (F1_URGENT - need.f1) * 100
            need.reasons.append(f"cross-session F1 {need.f1:.2f}")
        if need.n_sessions < MIN_SESSIONS:
            score += (MIN_SESSIONS - need.n_sessions) * 10
            need.reasons.append(
                f"only {need.n_sessions} session{'s' if need.n_sessions != 1 else ''}"
            )
        if need.spread > SPREAD_HIGH:
            score += (need.spread - SPREAD_HIGH) * 50
            need.reasons.append(f"high spread {need.spread:.3f}")
        if need.samples < target:
            score += (target - need.samples) / target * 5
            need.reasons.append(f"{need.samples}/{target} samples")

        need.priority = score
        needs.append(need)

    needs.sort(key=lambda n: (-n.priority, n.label))
    return needs


def print_plan(needs: list[ClassNeed], note: str | None, target: int) -> None:
    print(f"Recording plan  (target {target} samples, {MIN_SESSIONS}+ sessions per class)")
    print(f"F1 source: {note or 'NONE -- run python -m gestureflow.evaluate first; '
                                'prioritizing on spread and session coverage only'}")
    print()
    print(f"{'':>3} {'cls':>4} {'n':>5} {'sess':>5} {'spread':>7} {'xF1':>6}   why")
    print("-" * 78)
    for i, n in enumerate(needs, 1):
        if n.priority <= 0:
            continue
        f1s = f"{n.f1:.2f}" if n.f1 is not None else "  -"
        print(f"{i:>3} {n.label:>4} {n.samples:>5} {n.n_sessions:>5} "
              f"{n.spread:>7.3f} {f1s:>6}   {'; '.join(n.reasons)}")

    urgent = [n for n in needs if n.priority > 0]
    if not urgent:
        print("  every class meets the targets -- nothing to record")
        return

    print()
    print("Record in this order:", " ".join(n.label for n in urgent[:8]))
    print(f"\nNext:  python gestureflow/collect.py --label {urgent[0].label} "
          f"--session <new-session-name>")
    print("Use a session name you have not used before, and change something real between "
          "sittings:\nlighting, distance, time of day, camera angle. Two sittings an hour "
          "apart teach the model nothing.")


# ------------------------------------------------------------------ writing
class IncrementalWriter:
    """Appends each sample to the CSV as it is captured.

    collecter.py buffered every row in memory until a clean exit, so an interrupted
    sitting was lost entirely. Here a crash costs at most the current frame.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.n = 0
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=HEADER)
        if new_file:
            self._w.writeheader()
            self._fh.flush()

    def write(self, label: str, session: str, seq: int, pts: np.ndarray) -> None:
        self._w.writerow({
            "label": label,
            "session": session,
            "timestamp": time.time(),
            "seq": seq,
            **{c: float(v) for c, v in zip(COORD_COLS, pts.reshape(-1), strict=True)},
        })
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.n += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> IncrementalWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ capture
def capture(label: str, session: str, target: int, csv_path: Path) -> int:
    """Camera loop. Returns the number of samples written.

    Not exercised by the test suite -- it needs a physical webcam.
    """
    import cv2

    from gestureflow.detector import hand_detector

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera (index 0). Close other apps using it.")

    detector = hand_detector(max_hands=1)
    burst = False
    frame_i = 0

    existing = 0
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        existing = len(prev[(prev.label == label) & (prev.session == session)])
        if existing:
            print(f"({existing} already recorded for {label}/{session})")

    print(f"label={label}  session={session}  target={target}")
    print("s=save  c=burst  u=undo(in-memory only)  q=quit")

    with IncrementalWriter(csv_path) as writer:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_i += 1

                hands, _ = detector.findHands(frame, draw=False)
                pts = None
                if hands and hands[0].get("lmList"):
                    pts = np.asarray(hands[0]["lmList"], dtype=np.float32)[:, :2]

                display = frame.copy()
                if pts is not None:
                    for a, b in preprocessing.CONNECTIONS:
                        cv2.line(display, tuple(pts[a].astype(int)),
                                 tuple(pts[b].astype(int)), (200, 200, 200), 2)
                    for x, y in pts:
                        cv2.circle(display, (int(x), int(y)), 4, (0, 0, 255), -1)

                total = existing + writer.n
                colour = (0, 255, 0) if pts is not None else (0, 0, 255)
                cv2.putText(display, f"{label} / {session}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                cv2.putText(display, f"{total} / {target}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if burst:
                    cv2.circle(display, (display.shape[1] - 40, 40), 14, (0, 0, 255), -1)
                    cv2.putText(display, "REC", (display.shape[1] - 110, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if total >= target:
                    cv2.putText(display, "TARGET REACHED", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("collect", display)

                do_save = False
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("c"), ord("C")):
                    burst = not burst
                    print(f"burst {'ON' if burst else 'OFF'}")
                elif key in (ord("s"), ord("S")):
                    do_save = True

                if burst and pts is not None and frame_i % BURST_EVERY == 0:
                    do_save = True

                if do_save:
                    if pts is None:
                        print("no hand, nothing saved")
                    else:
                        writer.write(label, session, existing + writer.n, pts)
                        print(f"saved {existing + writer.n}", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted -- samples already written are safe")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return writer.n


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(LANDMARKS_CSV))
    ap.add_argument("--report", default=str(EVAL_REPORT))
    ap.add_argument("--plan", action="store_true",
                    help="print the recording plan and exit (no camera needed)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--auto", action="store_true",
                    help="walk the priority queue, highest-priority class first")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    f1, note = load_f1(Path(args.report))
    needs = build_plan(df, f1, args.target)

    if args.plan or (not args.label and not args.auto):
        print_plan(needs, note, args.target)
        return

    session = args.session or f"s{int(time.time())}"
    if session in set(df["session"].astype(str)):
        print(f"warning: session {session!r} already exists in the dataset. Reusing it "
              f"weakens cross-session evaluation -- prefer a new name.\n")

    queue = [args.label] if args.label else [n.label for n in needs if n.priority > 0]
    if not queue:
        print("nothing to record -- every class meets the targets")
        return

    for label in queue:
        print(f"\n=== {label} ===")
        n = capture(label, session, args.target, csv_path)
        print(f"wrote {n} rows for {label}/{session}")
        if args.auto and n == 0:
            print("stopping (no samples captured)")
            break

    print("\nRe-measure before trusting any number:\n"
          "  python -m gestureflow.evaluate --epochs 25")


if __name__ == "__main__":
    main()

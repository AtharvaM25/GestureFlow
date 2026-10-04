// Port of gestureflow/core/temporal.py for Edge mode: per-frame predictions -> committed
// letters. `pnpm test:edge` replays recorded streams through both and compares every frame.

export const CONF_SHOW = 0.7;
export const CONF_COMMIT = 0.9;
export const COMMIT_FRAMES = 15;
export const SMOOTH = 7;

export type Step = { smoothed: string | null; committed: string | null };

export class LetterCommitter {
  private recent: (string | null)[] = [];
  private stable = 0;

  get progress(): number {
    return this.stable / COMMIT_FRAMES;
  }

  update(label: string, conf: number): Step {
    this.push(conf > CONF_SHOW ? label : null);
    // Counter.most_common(1): highest count, ties go to the label seen first
    const counts = new Map<string, number>();
    for (const v of this.recent) if (v !== null) counts.set(v, (counts.get(v) ?? 0) + 1);
    let smoothed: string | null = null;
    let best = 0;
    for (const [k, n] of counts) {
      if (n > best) {
        best = n;
        smoothed = k;
      }
    }

    let committed: string | null = null;
    if (conf > CONF_COMMIT && smoothed === label) {
      this.stable += 1;
      if (this.stable >= COMMIT_FRAMES) {
        committed = label;
        this.stable = 0;
        this.recent = [];
      }
    } else {
      this.stable = 0;
    }
    return { smoothed, committed };
  }

  noHand() {
    this.push(null);
    this.stable = 0;
  }

  private push(v: string | null) {
    this.recent.push(v);
    if (this.recent.length > SMOOTH) this.recent.shift();
  }
}

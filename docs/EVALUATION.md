# Evaluation — measured, not quoted

> This is the original audit. A later re-run of the same unseen-session protocol on another
> machine gave **94.2% accuracy / 0.934 macro F1** (`docs/evaluation_report.json`); the README
> uses that run. The ~0.5-point difference is run-to-run variance.

Every number here was produced by running this repository's own code and data during the
audit of commit `6abad36`. Nothing is copied from the README. Reproduction commands are
given for each result.

---

## 1. Headline finding

**The repository's 97.6% test accuracy is inflated by session leakage. The honest
cross-session figure is 93.7%, and one class collapses almost completely.**

| Protocol | Train | Test | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| Repo `make_split` (tail of each class) | 2,052 rows, mixed | 455 rows, all `day2` | **0.9758** | 0.9744 |
| **Leave-one-session-out** (`day1` → `day2`) | 1,490 rows, `day1` only | 1,458 rows, `day2`, fully unseen | **0.9369** | 0.9275 |

Gap: **3.9 points of accuracy, 4.7 points of macro F1.**

The gap is the interesting part. It is small enough that the approach is clearly sound —
background-free skeleton rendering really does generalize across sittings — and large
enough that quoting 97.6% without the protocol would be misleading.

### Why the original split leaks

`CNN.make_split` sorts each class by `(session, timestamp)` and takes the tail:

```
train  n=2052  sessions={day1: 1490, day2: 562}
val    n=441   sessions={day2: 441}
test   n=455   sessions={day2: 455}
```

Validation and test are drawn **entirely** from `day2`, while `day2` also supplies 562 of
the 2,052 training rows — 27% of train, and 39% of all `day2` frames. `collecter.py`
burst-saves every 3rd frame of a continuous sitting, so a training frame and a test frame
can be ~0.1 s apart: near-duplicate hand poses on opposite sides of the split. The 0.9758
measures interpolation inside one recording session.

```bash
python -m gestureflow.splits   # prints the leak report above
```

---

## 2. The finding the leaky split was hiding

Per-class F1 under leave-one-session-out, worst first:

| Class | F1 (cross-session) | F1 (whole dataset, leaky) |
| --- | --- | --- |
| **K** | **0.18** | 0.957 |
| N | 0.66 | 0.927 |
| V | 0.69 | 0.957 |
| M | 0.84 | 0.963 |
| D | 0.92 | — |
| T | 0.93 | — |
| S | 0.94 | 0.955 |

**K does not survive an unseen session.** Under the contaminated split K looks healthy at
0.957; held out properly it scores 0.18, meaning the classifier gets it wrong most of the
time on a sitting it has not memorized. V (0.69) degrades alongside it, consistent with the
`K→V` confusion already visible in the leaky evaluation (10 cases).

This is a dataset and gesture-geometry problem, not a capacity problem — the model reaches
98.7% training accuracy in the same run. K and V differ mainly in thumb placement, which is
the landmark MediaPipe positions least reliably, and the author's K appears to have been
held differently across the two sittings.

**Actionable consequence:** re-record K (and V, N, M) across several sittings with
deliberate variation before any further architecture work. More epochs will not fix this.
This is exactly what the confusion-aware collection feature in the plan is for.

---

## 3. Measured model facts

| Property | Value | How measured |
| --- | --- | --- |
| Parameters | 586,234 | `sum(p.numel() for p in model.parameters())` |
| Checkpoint size | 2.37 MB | `os.path.getsize('models/gesture_cnn.pt')` |
| CNN latency, CPU, batch 1 | **19.08 ms** (52 fps) | 100 timed forwards after 10 warm-ups |
| Dataset | 2,948 samples, 26 classes, 2 sessions, 1 signer | `data/landmarks.csv` |
| Exact duplicate rows | 0 | coordinate-column duplicate check |

The latency figure is **CNN forward only**. It excludes MediaPipe landmark detection, which
dominates the real camera loop. End-to-end latency has not been measured and must not be
quoted until it is.

---

## 4. Protocol and caveats

- Cross-session run: train on `day1` (1,490 rows), test on `day2` (1,458 rows), 25 epochs,
  batch 64, Adam lr 1e-3, weight decay 1e-4, cosine schedule, label smoothing 0.05,
  `handnorm.augment` on train only — i.e. the repo's own recipe, changing only the split.
- The published checkpoint was trained for 40 epochs; this run used 25 for time. The
  cross-session figure may improve slightly with the full schedule.
- **Single run, single seed.** No variance estimate. Treat 0.9369 as a point estimate
  ±~1 point, not a precise value.
- Two sessions means leave-one-session-out gives only two folds, and only one was run
  (`day1 → day2`). The reverse fold is not reported because it was not measured.
- **All data comes from one signer.** Cross-signer generalization is completely unmeasured
  and no claim about it may be made. This is the largest open question about the project.

---

## 5. Claims permitted in the README today

**Permitted** (measured above): 586,234 parameters; 2.37 MB checkpoint; 19.08 ms CNN-only
CPU latency; 2,948 samples / 26 classes / 2 sessions / 1 signer; 0.9758 within-session;
**0.9369 cross-session**; K's cross-session collapse.

**Not permitted** until measured: end-to-end latency including MediaPipe; cross-signer
accuracy; browser/ONNX inference speed or parity; dynamic or two-hand gesture performance;
any agent or MCP capability not yet implemented.

**Rule:** the within-session number may only appear next to the cross-session number. On
its own it is misleading, and a reviewer who reruns the split will find out.

---

## 6. Reproduce

```bash
# leak report on both split strategies
python -m gestureflow.splits

# evaluate the shipped checkpoint on the repo's own split
python -c "import CNN"   # see docs/ for the audit scripts

# honest protocol: train day1, test day2
python -m gestureflow.evaluate --epochs 25
```

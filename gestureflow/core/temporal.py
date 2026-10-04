"""
Per-frame predictions -> committed letters.

Single frames flicker, so the displayed label is a majority vote over the last few
confident frames. A letter is committed only after the raw prediction agrees with that
vote at high confidence for COMMIT_FRAMES frames in a row.
"""

from collections import Counter, deque
from dataclasses import dataclass

CONF_SHOW = 0.70        # a frame votes only above this
CONF_COMMIT = 0.90      # a frame counts toward a commit only above this
COMMIT_FRAMES = 15      # consecutive agreeing frames needed to commit
SMOOTH = 7              # majority vote over this many recent frames


@dataclass
class Step:
    smoothed: str | None    # majority label over recent frames; None if no confident votes
    committed: str | None   # the letter committed on this frame, if any


class LetterCommitter:
    def __init__(self, conf_show=CONF_SHOW, conf_commit=CONF_COMMIT,
                 commit_frames=COMMIT_FRAMES, window=SMOOTH):
        self.conf_show = conf_show
        self.conf_commit = conf_commit
        self.commit_frames = commit_frames
        self.recent = deque(maxlen=window)
        self.stable = 0

    @property
    def progress(self):
        """How far the current hold is toward a commit, 0..1."""
        return self.stable / self.commit_frames

    def update(self, label, conf):
        """Feed one frame's prediction."""
        self.recent.append(label if conf > self.conf_show else None)
        votes = [v for v in self.recent if v is not None]
        smoothed = Counter(votes).most_common(1)[0][0] if votes else None

        committed = None
        if conf > self.conf_commit and smoothed == label:
            self.stable += 1
            if self.stable >= self.commit_frames:
                committed = label
                self.stable = 0
                self.recent.clear()
        else:
            self.stable = 0
        return Step(smoothed, committed)

    def no_hand(self):
        """Feed a frame with no hand detected."""
        self.recent.append(None)
        self.stable = 0

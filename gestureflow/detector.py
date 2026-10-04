"""cvzone's HandDetector, with its MediaPipe model kept in models/ instead of the current folder."""

from cvzone.HandTrackingModule import HandDetector

from gestureflow.paths import HAND_LANDMARKER


def hand_detector(max_hands=1):
    # cvzone downloads the model to HandDetector.MODEL_NAME, relative to the working
    # directory, on first use. Pointing it at models/ keeps one copy wherever you run from.
    HAND_LANDMARKER.parent.mkdir(exist_ok=True)
    HandDetector.MODEL_NAME = str(HAND_LANDMARKER)
    return HandDetector(maxHands=max_hands)

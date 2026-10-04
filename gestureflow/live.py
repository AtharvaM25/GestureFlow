"""
Live recognition from the webcam, then a sentence from the local LLM.
Run: python -m gestureflow.live

Camera, drawing and keys only. Recognition comes from gestureflow.core.session.
"""

import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv

from gestureflow.core.inference import ConfigMismatch, GesturePredictor
from gestureflow.core.session import RecognitionSession
from gestureflow.detector import hand_detector
from gestureflow.paths import MODEL_PATH
from gestureflow.sentence import generate_sentence

load_dotenv()


# ----------------------------------------------------------------- loop
def words_collector(cap, detector, predictor):
    sentence = []
    session = RecognitionSession(predictor)

    flash_until = 0.0

    print("q=quit  b=backspace")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # draw=False, then draw on a copy. Same discipline as collection.
        hands, _ = detector.findHands(frame, draw=False)
        display = frame.copy()

        if hands and hands[0].get("lmList"):
            x, y, w, h = hands[0]["bbox"]
            res = session.step(hands[0]["lmList"])

            colour = (0, 255, 0) if res.gesture else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), colour, 2)
            if res.gesture:
                cv2.putText(display, f"{res.gesture} {res.confidence:.2f}",
                            (x, max(30, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

            if res.committed:
                sentence.append(res.committed)
                print(f"committed {res.committed} -> {''.join(sentence)}")
                flash_until = time.time() + 1.0

            cv2.imshow("skeleton", cv2.resize(
                res.prediction.skeleton, (300, 300), interpolation=cv2.INTER_NEAREST))
        else:
            session.step(None)
            cv2.putText(display, "no hand", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if time.time() < flash_until:
            overlay = np.zeros_like(display)
            overlay[:, :, 1] = 255
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            cv2.putText(display, "LETTER ADDED", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(display, "".join(sentence), (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("gesture", display)

        raw = cv2.waitKey(1) & 0xFF
        ch = chr(raw).lower() if 32 <= raw < 127 else ""
        if ch == "q":
            break
        if ch == "b" and sentence:
            print(f"removed {sentence.pop()}")

    cap.release()
    cv2.destroyAllWindows()
    return sentence


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open camera")

    detector = hand_detector(max_hands=1)
    try:
        predictor = GesturePredictor(MODEL_PATH)
    except ConfigMismatch as e:
        sys.exit(f"ERROR: {e}")
    print(f"loaded {len(predictor.labels)} classes")

    letters = words_collector(cap, detector, predictor)
    print(letters)
    print(generate_sentence(letters))

"""
GestureFlow: landmark-based hand gesture recognition.

Programs (run from the project folder):

    python -m gestureflow.live       webcam recognition + sentence generation
    python -m gestureflow.collect    record landmarks, prioritised by what the model gets wrong
    python -m gestureflow.train      train the CNN
    python -m gestureflow.evaluate   accuracy on a session the model never saw
    python -m gestureflow.preview    mean skeleton per letter -> docs/images/preview_mean.png
"""

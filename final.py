import os
import cv2
import numpy as np
import tensorflow as tf
import subprocess
import threading

from collections import deque
from cvzone.HandTrackingModule import HandDetector

from hand_utils import load_class_names
from hand_utils import make_hand_canvas


# ---------------- SETTINGS ----------------

MODEL_PATH = "asl_best_model.keras"

LABELS_PATH = "class_names.txt"

DATA_DIR = "abc_data"

IMG_SIZE = 224

OFFSET = 20

CONFIDENCE_THRESHOLD = 0.30

DISPLAY_THRESHOLD = 0.30

WINDOW_NAME = "Image"

MAX_SENTENCE_LENGTH = 80

SMOOTHING_WINDOW = 8


# ---------------- SPEAK TEXT ----------------

def speak_text_async(text):

    # Do nothing if empty
    if not text.strip():
        return

    # Run speaking in background
    def _run():

        escaped = text.replace("'", "''")

        command = (
            "Add-Type -AssemblyName System.Speech; "
            f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak('{escaped}')"
        )

        subprocess.run(

            ["powershell", "-Command", command],

            stdout=subprocess.DEVNULL,

            stderr=subprocess.DEVNULL,

            check=False,
        )

    threading.Thread(
        target=_run,
        daemon=True
    ).start()


# ---------------- DRAW INFO BAR ----------------

def draw_sentence_bar(
    frame,
    sentence,
    saved_letter,
    status_text
):

    bar_height = 110

    frame_height, frame_width = frame.shape[:2]

    # Dark rectangle
    cv2.rectangle(

        frame,

        (0, frame_height - bar_height),

        (frame_width, frame_height),

        (25, 25, 25),

        cv2.FILLED,
    )

    # Saved letter
    cv2.putText(

        frame,

        f"Saved: {saved_letter or '-'}",

        (20, frame_height - 72),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2,
    )

    # Sentence
    cv2.putText(

        frame,

        f"Sentence: {sentence or '-'}",

        (20, frame_height - 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2,
    )

    # Status message
    cv2.putText(

        frame,

        status_text,

        (20, frame_height - 12),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (180, 255, 180),

        1,
    )


# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(f"Loaded model from {MODEL_PATH}")


# ---------------- LOAD LABELS ----------------

if os.path.exists(LABELS_PATH):

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = []

        for line in file:

            line = line.strip()

            if line:
                labels.append(line)

else:

    labels = load_class_names(DATA_DIR)

print(f"Loaded {len(labels)} labels: {labels}")

print("Press Q in the window to quit.")


# ---------------- VARIABLES ----------------

detector = HandDetector(maxHands=1)

sentence = ""

saved_letter = ""

status_text = (
    "Keys: S save letter, "
    "Space add gap, "
    "Backspace delete, "
    "C clear, "
    "Enter speak, "
    "Q quit"
)

current_prediction = ""

current_confidence = 0.0

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# ---------------- START CAMERA ----------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ---------------- MAIN LOOP ----------------

while True:

    # Read webcam frame
    success, img = cap.read()

    if not success:
        continue

    # Copy image
    imgOutput = img.copy()

    # Find hands
    hands, _ = detector.findHands(
        img,
        draw=False
    )

    # If hand found
    if hands:

        hand = hands[0]

        x, y, w, h = hand["bbox"]

        # Create white hand image
        imgWhite = make_hand_canvas(

            img,

            (x, y, w, h),

            img_size=IMG_SIZE,

            offset=OFFSET
        )

        # If hand image exists
        if imgWhite is not None:

            # Convert image into float
            img_input = imgWhite.astype(
                "float32"
            )

            # Normalize image
            img_input = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_input
            )

            # Add batch dimension
            img_input = np.expand_dims(
                img_input,
                axis=0
            )

            # Predict hand sign
            prediction = model.predict(
                img_input,
                verbose=0
            )[0]

            # Save predictions
            prediction_history.append(
                prediction
            )

            # Average predictions
            mean_predictions = np.mean(
                prediction_history,
                axis=0
            )

            # Best prediction index
            index = int(
                np.argmax(mean_predictions)
            )

            # Confidence score
            confidence = float(
                np.max(mean_predictions)
            )

            # Get label name
            if index < len(labels):
                label = labels[index]
            else:
                label = str(index)

            # Decide display text
            if confidence >= DISPLAY_THRESHOLD:
                display_text = label
            else:
                display_text = "Uncertain"

            current_prediction = label

            current_confidence = confidence

            # Top label box
            cv2.rectangle(

                imgOutput,

                (x, y - 50),

                (x + 280, y),

                (255, 0, 255),

                cv2.FILLED,
            )

            # Prediction text
            cv2.putText(

                imgOutput,

                f"{display_text} ({confidence:.2f})",

                (x, y - 15),

                cv2.FONT_HERSHEY_COMPLEX,

                0.9,

                (255, 255, 255),

                2,
            )

            # Hand rectangle
            cv2.rectangle(

                imgOutput,

                (x, y),

                (x + w, y + h),

                (255, 0, 255),

                2,
            )

            # Show processed hand
            cv2.imshow(
                "ProcessedHand",
                imgWhite
            )

    # If no hand found
    else:

        current_prediction = ""

        current_confidence = 0.0

        prediction_history.clear()

    # Draw sentence area
    draw_sentence_bar(

        imgOutput,

        sentence,

        saved_letter,

        status_text
    )

    # Show camera
    cv2.imshow(
        WINDOW_NAME,
        imgOutput
    )

    # Read keyboard
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):

        break

    # Save letter
    if key == ord("s"):

        if (
            current_prediction
            and current_confidence >= CONFIDENCE_THRESHOLD
        ):

            saved_letter = current_prediction

            if len(sentence) < MAX_SENTENCE_LENGTH:

                sentence += current_prediction

            status_text = (
                f"Saved '{current_prediction}'"
            )

        else:

            status_text = (
                "No confident prediction to save"
            )

    # Add space
    elif key == ord(" "):

        if sentence and not sentence.endswith(" "):

            sentence += " "

            status_text = "Added space"

    # Backspace
    elif key == 8:

        if sentence:

            sentence = sentence[:-1]

            status_text = (
                "Deleted last character"
            )

    # Clear sentence
    elif key == ord("c"):

        sentence = ""

        saved_letter = ""

        status_text = "Sentence cleared"

    # Speak sentence
    elif key == 13:

        if sentence.strip():

            status_text = (
                f"Speaking: {sentence}"
            )

            speak_text_async(sentence)

        else:

            status_text = "Sentence is empty"


# ---------------- CLOSE EVERYTHING ----------------

cap.release()

cv2.destroyAllWindows()
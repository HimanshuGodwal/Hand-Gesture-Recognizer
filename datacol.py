import cv2
import time

from cvzone.HandTrackingModule import HandDetector
from hand_utils import make_hand_canvas


# Start webcam
cap = cv2.VideoCapture(0)

# Detect one hand
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 224

# Folder where images will be saved
folder = "D:\\Me\\Semester\\SEM 4\\pblp\\ssss\\abc_data\\v"

counter = 0


while True:

    # Read camera frame
    success, img = cap.read()

    # Skip if camera fails
    if not success:
        continue

    # Detect hand
    hands, img = detector.findHands(img)

    # If hand found
    if hands:

        # Take first hand
        hand = hands[0]

        # Get hand box position
        x, y, w, h = hand['bbox']

        # Create white background hand image
        imgWhite = make_hand_canvas(
            img,
            (x, y, w, h),
            img_size=imgSize,
            offset=offset
        )

        # Show processed image
        if imgWhite is not None:
            cv2.imshow("ImageWhite", imgWhite)

    # Show webcam
    cv2.imshow("Image", img)

    # Wait for keyboard press
    key = cv2.waitKey(1)

    # Save image when "s" is pressed
    if key == ord("s") and hands and imgWhite is not None:

        counter += 1

        cv2.imwrite(
            f'{folder}/Image_{time.time()}.jpg',
            imgWhite
        )

        print(counter)
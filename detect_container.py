import cv2
import numpy as np


def main():
    capture()


def images():
    img = cv2.imread()


def capture():
    cap = cv2.VideoCapture(0)

    while True:
        # taking each frame
        ret, frame = cap.read()

        # converting BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # define range of blue color in HSV
        lower_blue = np.array([51, 5, 71])
        upper_blue = np.array([127, 255, 209])

        # Threshold the HSV frames to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
  
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

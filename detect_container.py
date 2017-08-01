import cv2

def main():
    capture()

def capture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

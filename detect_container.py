import cv2
import numpy as np
import tkinter as tk
import time
from threading import Timer


class MainApplication:
    def __init__(self, master):
        self.master = master
        master.title("ComputerVisionVolume")

        self.get_color_button = tk.Button(master, text="Press for color of cup", command=self.capture)
        self.get_color_button.pack()


    def detect_color_of_cup(self):
        """
        When the user presses 'q' a color value of the center pixel is returned in BGR.
        :return:
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            cv2.rectangle(frame, (590, 410), (710, 690), (0, 0, 0), 2)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # center of the rectangle, this pixel is read
        cup_pixel = frame[640, 540]

        print("The BGR of your cup is", cup_pixel)

        return cup_pixel

    def hsv_conversion(self):
        blue = np.uint8([[[255, 0, 0]]])
        hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
        print(hsv_blue)
        # blue = 120, 255, 255 = calculations( - 10, / 5.1, /5.1) lower(110, 50, 50) upper(130, 255, 255)


    def image_color_manipulation(self, original_image):

        lower_blue = np.array([39, 24, 121])
        upper_blue = np.array([179, 255, 255])

        hsv_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        smoothed = self.smoothing(mask)
        res = cv2.bitwise_and(hsv_img, hsv_img, mask=smoothed)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        return thresh, res


    def capture(self):
        capture = cv2.VideoCapture(0)


        while True:
            ret, frame = capture.read()

            #img = cv2.imread('images/bluecup2.jpg')
            binary_image, res = self.image_color_manipulation(frame)
            contour_outline = self.contours(binary_image, frame)
            cv2.imshow('video', contour_outline)


               # x, y = self.moments(contours)

                #self.draw_center(contour_outline, x, y)



            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


    def smoothing(self, res):
        median = cv2.medianBlur(res, 15)
        return median

    def contours(self, frame_threshed, original_frame):
        image, contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:

            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(original_frame, c, -1, (0, 0, 255), 3)

            x, y, w, h = cv2.boundingRect(c)

            cv2.rectangle(original_frame, (x,y), (x+w, y+h), (0,255, 0), 2)

            moments = cv2.moments(c)

            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])

            print('Centroid x', centroid_x, 'Centroid y', centroid_y)
            self.draw_center(original_frame, centroid_x, centroid_y)

        return original_frame


    def shape_math(self, contours):
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

    def draw_center(self, image, x, y):
        cv2.circle(image, (x, y), 10, (169, 34, 200), -1)








if __name__ == "__main__":
    root = tk.Tk()
    main_app = MainApplication(root)
    root.mainloop()

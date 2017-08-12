import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from scipy.spatial import distance as dist


KNOWN_HIEGHT = 11.0
KNOW_DISTANCE = 12.4 # jounral


class MainApplication:
    def __init__(self):

        self._root_window = tk.Tk()
        self._root_window.wm_title('Volume')

        # initialize image panel
        self.panel = tk.Label(self._root_window)
        self.panel.grid(row=2, column=1)

        self.get_color_button = tk.Button(master=self._root_window, text="New Container",
                                          command=self.key_shape)
        self.get_color_button.grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.N)

        self.cup_shape_button = tk.Button(master=self._root_window, text="New Color",
                                          command=self.key_color_grab)
        self.cup_shape_button.grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.N)

        self.run_button = tk.Button(master=self._root_window, text="Run",
                                          command=self.key_contours)
        self.run_button.grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.N)

        # define the lower and upper boundaries of the color in the HSV color space
        self.lower = {'cup': (39, 24, 121), 'card': (0, 147, 0)}  # assigning new item lower['card'] = (93, 2, 5)
        self.upper = {'cup': (179, 255, 255), 'card': (7, 255, 255)}

        # define color for outline of objects
        self.colors = {'cup': (0, 0, 255), 'card': (255,0,0)}

        self.height_inches_card = 2.125
        self.pixels_metric = None

        self.key = 3
        self.camera = cv2.VideoCapture(0)

    def run(self) -> None:
        self.show_frame()
        self._root_window.mainloop()

    def key_contours(self):
        self.key = 0

    def key_color_grab(self):
        self.key = 1

    def key_shape(self):
        self.key = 2

    def show_frame(self):
        grabbed, self.frame = self.camera.read()

        if grabbed:

            if self.key == 0:
                self.image_color_manipulation()
            elif self.key == 1:
                self.new_detect_color()
            elif self.key == 2:
                self.top_shape()
            else:
                pass
            # rearranging the color channel
            #self.image_color_manipulation()
            b, g, r = cv2.split(self.frame)
            self.frame = cv2.merge((r, g, b))
            img = Image.fromarray(self.frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
        self.panel.after(30, self.show_frame)

    def image_color_manipulation(self):
        blurred = cv2.GaussianBlur(self.frame, (11, 11), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for key, value in self.upper.items():
            # different methods for image manipulation

            kernel = np.ones((9, 9), np.uint8)

            # more lighter and just as accurate
            mask = cv2.inRange(hsv_img, self.lower[key], self.upper[key])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # this. bit more accurate but very power intensive
            # smooth = self.smoothing(mask)
            # res = cv2.bitwise_and(hsv_img, hsv_img, mask=smooth)
            # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            # thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            self.contours(mask, key)

    def contours(self, mask, key):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            #cv2.drawContours(self.frame, c, -1, (0, 0, 255), 3)

            moments = cv2.moments(c)

            #x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])

            # print('Centroid x', centroid_x, 'Centroid y', centroid_y)
            self.draw_center(self.frame, centroid_x, centroid_y)
            self.pixels_per_metric(c)

            #cv2.putText(self.frame, key, (centroid_x, centroid_y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[key],2)

    def new_detect_color(self):

        cv2.rectangle(self.frame, (590, 410), (710, 690), (0, 0, 0), 2)
        cup_pixel = self.frame[640, 540]
        print("The BGR of your cup is", cup_pixel)
        return cup_pixel

    def top_shape(self):
        pass
        print("show top of container.")

    def midpoint(self, ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def pixels_per_metric(self, cnts):
        box = cv2.minAreaRect(cnts)
        box_points = cv2.boxPoints(box)
        box = np.array(box_points, dtype="int")

        cv2.drawContours(self.frame, [box.astype("int")], -1, (0, 0, 100), 2)
        for (x, y) in box_points:
            cv2.circle(self.frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        (bottom_left, top_left, top_right, bottom_right) = box

        (tltrX, tltrY) = self.midpoint(top_left, top_right)
        (blbrX, blbrY) = self.midpoint(bottom_left, bottom_right)

        (tlblX, tlblY) = self.midpoint(top_left, bottom_left)
        (trbrX, trbrY) = self.midpoint(top_right, bottom_right)

        # draw the midpoints on the image
        cv2.circle(self.frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(self.frame, (int(blbrX), int(blbrY)), 5, (0, 255, 0), -1)
        cv2.circle(self.frame, (int(tlblX), int(tlblY)), 5, (0, 0, 255), -1)
        cv2.circle(self.frame, (int(trbrX), int(trbrY)), 5, (0, 0, 0), -1)

        cv2.line(self.frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(self.frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        height_pixels = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        width_pixels = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


        if self.pixels_metric is None:
            self.pixels_metric = height_pixels / self.height_inches_card

        print("height in pixels =", height_pixels, " and width =", width_pixels)
        # height in pixels = 377.8160928282436  and width = 206.9685966517626

        # compute the size of the object
        dimension_inchesA = height_pixels / self.pixels_metric
        dimension_inchesB = width_pixels / self.pixels_metric

        # draw the object sizes on the image
        cv2.putText(self.frame, "{:.1f}in".format(dimension_inchesA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        cv2.putText(self.frame, "{:.1f}in".format(dimension_inchesB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)


    def smoothing(self, res):
        median = cv2.medianBlur(res, 15)
        return median

    def draw_center(self, image, x, y):
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)

    def distance_to_camera(self, knowWidth, focallength, perWidth):
        return (knowWidth * focallength) / perWidth


if __name__ == "__main__":
    MainApplication().run()

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


KNOWN_HIEGHT = 11.0
KNOW_DISTANCE = 12.4 # jounral


class MainApplication:
    def __init__(self):

        self._root_window = tk.Tk()
        self._root_window.wm_title('Volume')

        #self.imageFrame = tk.Frame(master=self._root_window, width=600, height=500)
        #self.imageFrame.grid(row=1, column=1, padx=10, pady=2)

        # initialize image panel
        self.panel = tk.Label(self._root_window)
        self.panel.grid(row=2, column=1)

        self.camera = cv2.VideoCapture(0)

        self.get_color_button = tk.Button(master=self._root_window, text="New Color",
                                          command=self.detect_color_of_cup)
        self.get_color_button.grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.N)

        self.cup_shape_button = tk.Button(master=self._root_window, text="New Shape",
                                          command=self.top_shape)
        self.cup_shape_button.grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.N)

        self.run_button = tk.Button(master=self._root_window, text="Run",
                                          command=self.show_frame)
        self.run_button.grid(
            row=0, column=2, padx=5, pady=5, sticky=tk.N)


        # define the lower and upper boundaries of the color in the HSV color space
        self.lower = {'cup': (39, 24, 121), 'card': (0, 147, 0)} # assigning new item lower['card'] = (93, 2, 5)
        self.upper = {'cup': (179, 255, 255), 'card': (7, 255, 255)}

        # define color for outline of objects
        self.colors = {'cup': (0, 0, 255), 'card': (255,0,0)}



    def run(self) -> None:
        self._root_window.mainloop()


    def open_camera(self):
        self.camera = cv2.VideoCapture(0)

    def show_frame(self):

        grabbed, self.frame = self.camera.read()

        if grabbed:

            self.image_color_manipulation()

            img = Image.fromarray(self.frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk

            self.panel.configure(image=imgtk)
        self.panel.after(30, self.show_frame)


    def capture(self):
        camera = cv2.VideoCapture(0)

        while True:
            grabbed, self.frame = camera.read()
            self.image_color_manipulation()

            cv2.imshow('video', self.frame)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

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

        # rearranging the color channel
        b, g, r = cv2.split(self.frame)
        self.frame = cv2.merge((r, g, b))




    def contours(self, mask, key):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(self.frame, c, -1, (0, 0, 255), 3)

            moments = cv2.moments(c)

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            area = cv2.contourArea(c)

            # print('Centroid x', centroid_x, 'Centroid y', centroid_y)
            self.draw_center(self.frame, centroid_x, centroid_y)

            cv2.putText(self.frame, key, (centroid_x, centroid_y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[key],2)


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
        cup_pixel = frame[640, 540]

        print("The BGR of your cup is", cup_pixel)
        return cup_pixel

    def top_shape(self):
        pass


    def smoothing(self, res):
        median = cv2.medianBlur(res, 15)
        return median

    def draw_center(self, image, x, y):
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1)

    def distance_to_camera(self, knowWidth, focallength, perWidth):
        return (knowWidth * focallength) / perWidth


if __name__ == "__main__":
    MainApplication().run()

# coding:utf-8
import os
import shutil
import math
import cv2
import numpy as np
class Logger(object):
    def __init__(self, save_dir, header=""):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        self.save_dir = save_dir
        self.log_txt = header +" \n"
    def log(self, values):
        self.log_txt += ",".join(map(str, values)) + "\n"
    def close(self):
        with open(os.path.join(self.save_dir, "log.txt"), "w") as f:
            f.write(self.log_txt)

def render_cv2img(pos, deg, save_path=None):
    # left, top = 0,0
    deg *= 60
    pole_len = 100
    lane_pos = 250
    cart_x = int(300 + (300 * (pos*100 / 240)))
    cart_y = lane_pos
    tip_x = cart_x + round(pole_len * math.sin(math.radians(deg)))
    tip_y = cart_y - round(pole_len * math.cos(math.radians(deg)))
    img = np.zeros((400, 600, 3), np.uint8)
    img = cv2.line(img, (0,lane_pos), (600,lane_pos), (255,255,255), 1)
    img = cv2.line(img, (cart_x, cart_y), (tip_x, tip_y), (255,255,255), 5)
    img = cv2.circle(img, (cart_x, cart_y), 10, (127,127,127), 3)
    if save_path is not None:
        saved = cv2.imwrite(save_path, img)
        if not saved:
            print("img not saved")
    return img

class VideoRecorder(object):
    def __init__(self, save_path):
        fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
        self.out = cv2.VideoWriter(save_path, fourcc, 20, (600,400))
    # def __init__(self):
    #     pass
    def record(self, pos, deg):
        self.out.write(self._render_cv2img(pos, deg))
    def save(self):
        self.out.release()
    def _render_cv2img(self, pos, deg):
        # left, top = 0,0
        deg *= 60
        pole_len = 100
        lane_pos = 250
        cart_x = int(300 + (300 * (pos*100 / 240)))
        cart_y = lane_pos
        tip_x = cart_x + round(pole_len * math.sin(math.radians(deg)))
        tip_y = cart_y - round(pole_len * math.cos(math.radians(deg)))
        img = np.zeros((400, 600, 3), np.uint8)
        img = cv2.line(img, (0,lane_pos), (600,lane_pos), (255,255,255), 1)
        img = cv2.line(img, (cart_x, cart_y), (tip_x, tip_y), (255,255,255), 5)
        img = cv2.circle(img, (cart_x, cart_y), 10, (127,127,127), 3)
        return img

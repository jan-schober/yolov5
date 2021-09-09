import cv2
import numpy as np
from PIL import Image
import glob

def convert_optical_flow(flow, frame):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid



def main():
    img_list = sorted(glob.glob('/home/schober/yolov5/farneback_test/new_small/*.png'))
    for i in range(0, len(img_list)-1):
        img_0 = cv2.imread(img_list[i])
        img_0_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
        img_1 = cv2.imread(img_list[i+1])
        img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        OF_Farneback = cv2.calcOpticalFlowFarneback(img_0_gray, img_1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        OF_vis = Image.fromarray(convert_optical_flow(OF_Farneback, img_0))

        grid_1 = image_grid([Image.fromarray(img_0_gray), Image.fromarray(img_1_gray), OF_vis], rows=1, cols=3)
        grid_1.save('/home/schober/yolov5/farneback_test/new_small/out/of_' + str(i)+'_' + str(i+1) + '.png')

if __name__ == "__main__":
    main()

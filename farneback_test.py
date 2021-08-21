import cv2
import numpy as np
from PIL import Image

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
    img_0 = cv2.imread('/home/schober/yolov5/farneback_test/rendered_3.jpg')
    img_0_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    img_1 = cv2.imread('/home/schober/yolov5/farneback_test/rendered_2.jpg')
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

    """
    img_2 = cv2.imread('/home/schober/yolov5/farneback_test/2.png')
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    img_3 = cv2.imread('/home/schober/yolov5/farneback_test/3.png')
    img_3_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
    """
    img_rgb = cv2.imread('/home/schober/yolov5/farneback_test/carla0301_000324_leftImg8bit.png')


    OF_0_1 = cv2.calcOpticalFlowFarneback(img_0_gray, img_1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #OF_1_2 = cv2.calcOpticalFlowFarneback(img_1_gray, img_2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #OF_2_3 = cv2.calcOpticalFlowFarneback(img_2_gray, img_3_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    tof_0_1 = np.mean(np.sqrt(np.sum(OF_0_1 * OF_0_1, axis=-1)))
    print('OF_0_1', str(tof_0_1))
    #tof_1_2 = np.mean(np.sqrt(np.sum(OF_1_2 * OF_1_2, axis=-1)))
    #print('OF_1_2', str(tof_1_2))
    #tof_2_3 = np.mean(np.sqrt(np.sum(OF_2_3 * OF_2_3, axis=-1)))
    #print('OF_2_3', str(tof_2_3))


    rgb_1 = Image.fromarray(convert_optical_flow(OF_0_1, img_rgb))
    #rgb_2 = Image.fromarray(convert_optical_flow(OF_1_2, img_rgb))
    #rgb_3 = Image.fromarray(convert_optical_flow(OF_2_3, img_rgb))

    img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR)
    img_0 = Image.fromarray(img_0)
    img_1 = Image.fromarray(img_1)
    #img_2 = Image.fromarray(img_2)
    #img_3 = Image.fromarray(img_3)

    grid_1 = image_grid([img_0, img_1, rgb_1], rows=1, cols=3)
    #grid_2 = image_grid([img_1, img_2, rgb_2], rows=1, cols=3)
    #grid_3 = image_grid([img_2, img_3, rgb_3], rows=1, cols=3)

    grid_1.save('/home/schober/yolov5/farneback_test/rendered_of_1.png')
    #grid_2.save('/home/schober/yolov5/farneback_test/OF_2_geo.png')
    #grid_3.save('/home/schober/yolov5/farneback_test/OF_3_geo.png')
if __name__ == "__main__":
    main()

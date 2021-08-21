import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw

rendered_list = sorted(glob.glob('/home/schober/carla/output_cityscapes/images/val_rendered/*.jpg'))
augmented_list = sorted(glob.glob('/home/schober/carla/output_cityscapes/images/val_augmented/*.png'))

labels_augmented = sorted(glob.glob('/home/schober/yolov5/runs/test/carla_augmented2/labels/*.txt'))
labels_rendered = sorted(glob.glob('/home/schober/yolov5/runs/test/carla_rendered2/labels/*.txt'))
labels_gt = sorted(glob.glob('/home/schober/carla/output_cityscapes/resized/labels/*txt'))

print(len(rendered_list))
print(len(augmented_list))
print(len(labels_augmented))
print(len(labels_rendered))
print(len(labels_gt))

assert len(rendered_list) == len(augmented_list) ==  len(labels_augmented) == len(labels_rendered) == len(labels_gt), "Different Lenght of lists"

video_name = 'carla_yolov5_3.avi'

#black_image = Image.new('RGB', (512, 256), (0, 0, 0))


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def draw_rectangles(IMG, x_arr, y_arr, w_arr, h_arr):
    if len(x_arr) != 0:
        for x, y, w, h in zip(x_arr, y_arr, w_arr, h_arr):
            image_width, image_height = IMG.size

            x_0 = (x * image_width) - (w *image_width)/2
            x_1 = (x * image_width) + (w *image_width)/2

            y_0 = (y *image_height) - (h*image_height)/2
            y_1 = (y *image_height) + (h*image_height)/2
            img_with_rectangle = ImageDraw.Draw(IMG)
            img_with_rectangle.rectangle([(x_0, y_0), (x_1, y_1)],outline = (255, 0 , 0),  width = 1)
        return IMG
    else:
        return IMG

def convert_txt_arr(content_txt):
    x_arr = []
    y_arr = []
    w_arr = []
    h_arr = []
    for line in content_txt:
        cls = int(line.split()[0])
        if cls == 2:
            conf = float(line.split()[5])
            if conf >= 0.5:
                x_arr.append(float(line.split()[1]))
                y_arr.append(float(line.split()[2]))
                w_arr.append(float(line.split()[3]))
                h_arr.append(float(line.split()[4]))
    return x_arr, y_arr, w_arr, h_arr


def convert_txt_gt_arr(content_txt):
    x_arr = []
    y_arr = []
    w_arr = []
    h_arr = []
    for line in content_txt:
        cls = int(line.split()[0])
        if cls == 2:
            x_arr.append(float(line.split()[1]))
            y_arr.append(float(line.split()[2]))
            w_arr.append(float(line.split()[3]))
            h_arr.append(float(line.split()[4]))
    return x_arr, y_arr, w_arr, h_arr

first_list = [rendered_list[0], rendered_list[0], augmented_list[0], augmented_list[0]]
image, *images = [Image.open(file) for file in first_list]
example_grid = image_grid([image, *images], rows=2, cols=2)
#example_grid.save('example_grid.png')

width, height = example_grid.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for rendered_path, augmented_path, label_ren_path, label_aug_path, labels_gt_path in zip(rendered_list[::5], augmented_list[::5], labels_rendered[::5], labels_augmented[::5], labels_gt[::5]):
    with open(labels_gt_path) as f:
        content_aug = f.readlines()
        gt_x, gt_y, gt_w, gt_h = convert_txt_gt_arr(content_aug)
        f.close()
    with open(label_aug_path) as f:
        content_aug = f.readlines()
        aug_x, aug_y, aug_w, aug_h = convert_txt_arr(content_aug)
        f.close()
    with open(label_ren_path) as f:
        content_ren = f.readlines()
        rend_x, rend_y, rend_w, rend_h = convert_txt_arr(content_ren)
        f.close()

    rend_gt_img = draw_rectangles(Image.open(rendered_path).convert('RGB'), gt_x, gt_y, gt_w, gt_h)
    aug_gt_img = draw_rectangles(Image.open(augmented_path).convert('RGB'), gt_x, gt_y, gt_w, gt_h)
    rend_pred_img = draw_rectangles(Image.open(rendered_path).convert('RGB'), rend_x, rend_y, rend_w, rend_h)
    aug_pred_img = draw_rectangles(Image.open(augmented_path).convert('RGB'), aug_x, aug_y, aug_w, aug_h)

    #image, *images = [Image.open(file) for file in file_list]

    grid = image_grid([rend_pred_img, aug_pred_img, rend_gt_img, aug_gt_img], rows=2, cols=2)
    grid_cv = np.array(grid)
    grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
    video.write(grid_cv)
video.release()

import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw

real_folder = '/home/schober/vkitti_conv/images/val_real/'
real_list = sorted(glob.glob(real_folder+ '*.png'))
rendered_folder = '/home/schober/vkitti_conv/images/val_rendered/'
rendered_list = sorted(glob.glob(rendered_folder + '*.jpg'))
augmented_folder = '/home/schober/vkitti_conv/images/val_augmented/'
augmented_list = sorted(glob.glob(augmented_folder + '*.png'))

labels_aug_folder = '/home/schober/yolov5/runs/test/vkitti_augmented3/labels/'
labels_augmented = sorted(glob.glob(labels_aug_folder + '*.txt'))
labels_ren_folder = '/home/schober/yolov5/runs/test/vkitti_rendered3/labels/'
labels_rendered = sorted(glob.glob(labels_ren_folder + '*.txt'))
labels_real_folder = '/home/schober/yolov5/runs/test/vkitti_real3/labels/'
labels_real = sorted(glob.glob(labels_real_folder + '*.txt'))
labels_gt_folder ='/home/schober/vkitti_conv/labels/val/'
labels_gt = sorted(glob.glob(labels_gt_folder + '*.txt'))
labels_gt_real_folder ='/home/schober/vkitti_conv/bbox_kitti/darknet/'
labels_gt_real = sorted(glob.glob(labels_gt_folder + '*.txt'))


print(len(real_list))
print(len(rendered_list))
print(len(augmented_list))
print(len(labels_augmented))
print(len(labels_rendered))
print(len(labels_gt))
print(labels_gt_real)

set_real    =   set([x.split('/')[-1].replace('.png', '') for x in real_list])
set_ren     =   set([x.split('/')[-1].replace('.jpg', '') for x in rendered_list])
set_aug     =   set([x.split('/')[-1].replace('.png', '') for x in augmented_list])
set_lab_real=   set([x.split('/')[-1].replace('.txt', '') for x in labels_real])
set_lab_ren =   set([x.split('/')[-1].replace('.txt', '') for x in labels_rendered])
set_lab_aug =   set([x.split('/')[-1].replace('.txt', '') for x in labels_augmented])
set_lab_gt  =   set([x.split('/')[-1].replace('.txt', '') for x in labels_gt])
set_lab_gt_real = set([x.split('/')[-1].replace('.txt', '') for x in labels_gt_real])

print(set_real)
intersection = sorted(set.intersection(set_real, set_ren, set_aug, set_lab_real, set_lab_ren, set_lab_aug, set_lab_gt,set_lab_gt_real))

print(intersection)

real_list_up = []
rend_list_up = []
aug_list_up = []
lab_real_up = []
lab_rend_up = []
lab_aug_up = []
lab_gt_up = []
lab_gt_real_up = []
for file in intersection:
    real_list_up.append(real_folder + file + '.png' )
    rend_list_up.append(rendered_folder + file +'.jpg')
    aug_list_up.append(augmented_folder + file + '.png')
    lab_real_up.append(labels_real_folder + file + '.txt')
    lab_rend_up.append(labels_ren_folder+ file + '.txt')
    lab_aug_up.append(labels_aug_folder+ file + '.txt')
    lab_gt_up.append(labels_gt_folder+ file + '.txt')
    lab_gt_real_up.append(labels_gt_real_folder+ file + '.txt')
assert len(real_list_up) == len(rend_list_up) == len(aug_list_up) ==  len(lab_real_up) == len(lab_rend_up) == len(lab_aug_up) == len(lab_gt_up) == len(lab_gt_real_up), "Different Lenght of lists"

video_name = 'vkitti_yolov5_3.avi'

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

first_list = [real_list_up[0], rend_list_up[0], aug_list_up[0], real_list_up[0], rend_list_up[0], aug_list_up[0]]
image, *images = [Image.open(file) for file in first_list]
example_grid = image_grid([image, *images], rows=2, cols=3)
#example_grid.save('example_grid.png')

width, height = example_grid.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

for real_path, rendered_path, augmented_path, label_real_path, label_ren_path, label_aug_path, labels_gt_path, labels_gt_real_path in zip(real_list_up[::5], rend_list_up[::5], aug_list_up[::5], lab_real_up[::5], lab_rend_up[::5], lab_aug_up[::5], lab_gt_up[::5],lab_gt_real_up[::5]):

    with open(label_real_path) as f:
        content_aug = f.readlines()
        rl_x, rl_y, rl_w, rl_h = convert_txt_arr(content_aug)
        f.close()
    with open(labels_gt_path) as f:
        content_aug = f.readlines()
        gt_x, gt_y, gt_w, gt_h = convert_txt_gt_arr(content_aug)
        f.close()
    with open(labels_gt_real_path) as f:
        content_aug = f.readlines()
        gt_real_x, gt_real_y, gt_real_w, gt_real_h = convert_txt_gt_arr(content_aug)
        f.close()
    with open(label_aug_path) as f:
        content_aug = f.readlines()
        aug_x, aug_y, aug_w, aug_h = convert_txt_arr(content_aug)
        f.close()
    with open(label_ren_path) as f:
        content_ren = f.readlines()
        rend_x, rend_y, rend_w, rend_h = convert_txt_arr(content_ren)
        f.close()
    real_gt_img = draw_rectangles(Image.open(real_path).convert('RGB'), gt_real_x, gt_real_y, gt_real_w, gt_real_h)
    rend_gt_img = draw_rectangles(Image.open(rendered_path).convert('RGB'), gt_x, gt_y, gt_w, gt_h)
    aug_gt_img = draw_rectangles(Image.open(augmented_path).convert('RGB'), gt_x, gt_y, gt_w, gt_h)

    real_pred_img = draw_rectangles(Image.open(real_path).convert('RGB'), rl_x, rl_y, rl_w, rl_h)
    rend_pred_img = draw_rectangles(Image.open(rendered_path).convert('RGB'), rend_x, rend_y, rend_w, rend_h)
    aug_pred_img = draw_rectangles(Image.open(augmented_path).convert('RGB'), aug_x, aug_y, aug_w, aug_h)

    #image, *images = [Image.open(file) for file in file_list]

    grid = image_grid([real_pred_img, rend_pred_img, aug_pred_img, real_gt_img, rend_gt_img, aug_gt_img], rows=2, cols=3)
    grid_cv = np.array(grid)
    grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
    video.write(grid_cv)
video.release()

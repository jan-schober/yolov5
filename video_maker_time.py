import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

segmented_list = sorted(glob.glob('/home/schober/carla/for_yolov5/images/carla_rendered_01/*.jpg'))
augmented_list = sorted(glob.glob('/home/schober/carla/for_yolov5/images/carla_01_fest_z/*.png'))
gt_flow_list = sorted(glob.glob('output_time_final/*gt.png'))
ag_flow_list= sorted(glob.glob('output_time_final/*ag.png'))
diff_flow_list= sorted(glob.glob('output_time_final/*diff.png'))


print(len(segmented_list))
print(len(gt_flow_list))

assert len(segmented_list) ==  len(augmented_list) == len(gt_flow_list) == len(ag_flow_list) == len(diff_flow_list),  "Different Lenght of lists"

video_name = 'time_consistency_final.avi'

black_image = Image.new('RGB', (512, 256), (0, 0, 0))


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

first_list = [segmented_list[0], augmented_list[0], gt_flow_list[0], ag_flow_list[0]]
image, *images = [Image.open(file) for file in first_list]
example_grid = image_grid([image, *images], rows=2, cols=2)
#example_grid.save('example_grid.png')

width, height = example_grid.size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))


tof_arr = np.load('tof_arr_final.npy')

for segmented_path, augmented_path, gt_flow_path, ag_flow_path, diff_flow_path, tofl in zip(segmented_list, augmented_list, gt_flow_list, ag_flow_list, diff_flow_list, tof_arr):

    segmented_img = Image.open(segmented_path)
    augmented_img = Image.open(augmented_path)
    gt_flow_img = Image.open(gt_flow_path)
    ag_flow_img = Image.open(ag_flow_path)
    diff_flow_img = Image.open(diff_flow_path)
    diff_flow_draw = ImageDraw.Draw(diff_flow_img)
    font = ImageFont.load_default()
    diff_flow_draw.text((0, 0), str(tofl), (255, 255, 255), font=font)

    grid = image_grid([segmented_img,  augmented_img, gt_flow_img, ag_flow_img], rows=2, cols=2)
    grid_cv = np.array(grid)
    grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
    video.write(grid_cv)
video.release()

import glob
import cv2
import numpy as np
from PIL import Image

folder_list = ['01', '01_01', '01_02', '03', '03_01', '03_02', '04', '05']

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main():
    for folder in folder_list:
        augmented_list = sorted(glob.glob('/home/schober/yolov5/runs/detect/augmented_img/' + folder +'/*.png'))
        rendered_list = sorted(glob.glob('/home/schober/yolov5/runs/detect/rendered_img/' + folder +'/*.jpg'))

        video_name = 'carla_yolov_'+folder + '.avi'
        ex_0 = Image.open(augmented_list[0])
        ex_1 = Image.open(rendered_list[0])

        example_grid = image_grid([ex_0, ex_1],1,2)
        width, height = example_grid.size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))



        for aug_file, rend_file in zip(augmented_list, rendered_list):
            aug_img = Image.open(aug_file)
            rend_img = Image.open(rend_file)
            grid = image_grid([aug_img, rend_img], 1,2)

            grid_cv = np.array(grid)
            grid_cv = cv2.cvtColor(grid_cv, cv2.COLOR_RGB2BGR)
            video.write(grid_cv)
        video.release()
if __name__ == "__main__":
    main()

import glob
import pandas as pd

root_path_kitti = '/home/schober/vkitti_conv/bbox_vkitti/'

list_bboxes = sorted(glob.glob(root_path_kitti + '*/clone/bbox.txt'))
list_infos = sorted(glob.glob(root_path_kitti + '*/clone/info.txt'))
output_root = '/home/schober/vkitti_conv/labels/val/'
image_width = 1242
image_height = 375

for bbox_path, info_path in zip(list_bboxes, list_infos):
    with open(bbox_path) as f:
        next(f)
        content = f.readlines()
    info_pd = pd.read_csv(info_path, delimiter=' ', header=0)
    scene = bbox_path.split('/')[5]
    for line in content:
        frame = int(line.split()[0])
        frame_output = "{:06d}".format(frame)
        with open(output_root + scene + '_' + str(frame_output) + '.txt', 'a') as output:
            camera_id = int(line.split()[1])
            if camera_id == 0:
                occluded = float(line.split()[9])
                if occluded >= 0.25:
                    track_id = int(line.split()[2])
                    sub_df = info_pd[(info_pd['trackID']==track_id)]
                    object_class = sub_df.iloc[0]['label']
                    if object_class == 'Car' or object_class == 'Van':
                        object_id = 2
                    elif object_class == 'Truck':
                        object_id = 7
                    else:
                        assert 'object id not defined'
                    left = int(line.split()[3])
                    right = int(line.split()[4])

                    top = int(line.split()[5])
                    bot = int(line.split()[6])

                    x_center = (left + ((right - left) / 2)) / image_width
                    y_center = (top + ((bot - top) / 2)) / image_height

                    width = (right - left) / image_width
                    height = (bot - top) / image_height
                    output.write(str(object_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
        output.close()

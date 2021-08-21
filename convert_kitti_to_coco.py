import glob

kitti_bb = sorted(glob.glob('/home/schober/vkitti_conv/bbox_kitti/'+'*.txt'))
output_root = '/home/schober/vkitti_conv/labels/val/'
image_width = 1242
image_height = 375

for file in kitti_bb:
    with open(file) as f:
        content = f.readlines()
    scene = file.split('/')[-1]
    scene = scene.split('.')[0]
    for line in content:
        frame = int(line.split()[0])
        frame_output = "{:06d}".format(frame)
        with open(output_root + scene + '_' + str(frame_output) + '.txt', 'a') as output:
            occluded = int(line.split()[4])
            if occluded <= 2:
                object_class = line.split()[2]
                if object_class == 'Car' or object_class == 'Van':
                    object_id = 2
                    left = float(line.split()[6])
                    top = float(line.split()[7])
                    right = float(line.split()[8])
                    bot = float(line.split()[9])
                    x_center = (left + ((right - left) / 2)) / image_width
                    y_center = (top + ((bot - top) / 2)) / image_height
                    width = (right - left) / image_width
                    height = (bot - top) / image_height
                    output.write(
                        str(object_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(
                            height) + '\n')
                elif object_class == 'Truck':
                    object_id = 7
                    print('TRUUCK')
                    left = float(line.split()[6])
                    top = float(line.split()[7])
                    right = float(line.split()[8])
                    bot = float(line.split()[9])
                    x_center = (left + ((right - left) / 2)) / image_width
                    y_center = (top + ((bot - top) / 2)) / image_height
                    width = (right - left) / image_width
                    height = (bot - top) / image_height
                    output.write(
                        str(object_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(
                            height) + '\n')
                else:
                    pass

        output.close()


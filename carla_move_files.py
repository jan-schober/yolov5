import glob
from shutil import copyfile

town = 'output_carla_town_05'
prefix ='carla05_'

root_path = '/home/schober/carla/'+town+'/darknet/data/obj/'
out_folder_image = '/home/schober/carla/for_yolov5/images/val_rendered/'
out_folder_label = '/home/schober/carla/for_yolov5/labels/val/'
images = sorted(glob.glob(root_path + '*.jpg'))
labels = sorted(glob.glob('/home/schober/carla/'+town+'/labels_resized_new/' + '*.txt'))

def get_carla_number(path):
    file_name = path.split('/')[-1]
    number = file_name.split('.')[0]
    return number
'''
for path in images:
    img_number = get_carla_number(path)
    output_dst = out_folder_image + 'carla05_' + str(img_number) + '_leftImg8bit.jpg'
    copyfile(path, output_dst)
'''
for path in labels:
    img_number = get_carla_number(path)
    output_dst = out_folder_label + prefix + str(img_number) + '_leftImg8bit.txt'
    copyfile(path, output_dst)


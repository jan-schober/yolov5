import glob
import os

files = glob.glob('/home/schober/carla/output_cityscapes_new/images/val/*')
root_path = '/home/schober/carla/output_cityscapes_new/images/val/'
for file in files:
    file_name = file.split('/')[-1]
    file_name = file_name.replace('carla_', '')
    file_name = file_name.replace('_leftImg8bit', '')
    '''
    file_name = file_name.split('.')[0]
    number = int(file_name.split('_')[1])
    number = "{:06d}".format(number)
    file_name_new = file_name.split('_')[0] + '_'+str(number)+'.png'
    new_file = os.path.join(root_path, file_name_new)
    '''
    new_file = os.path.join(root_path, file_name)
    os.rename(file, new_file)



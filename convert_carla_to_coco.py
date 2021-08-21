import glob

root_path = '/home/schober/carla/output_cityscapes/darknet/data/obj/'
output_path = '/home/schober/carla/output_cityscapes/labels/val/'

labels_input = glob.glob(root_path + '*.txt')

for label in labels_input:
    with open(label) as f:
        content = f.readlines()
    data_str = ''
    file_name = label.split('/')[-1]
    with open(output_path + file_name, 'a') as outfile:
        for line in content:
            input_class = int(line.split()[0])
            if input_class == 0:
                output_class = 2
            elif input_class == 1:
                output_class = 7
            elif input_class == 2:
                output_class = 3
            elif input_class == 3:
                output_class = 1
            elif input_class == 4:
                output_class = 0
            x_center = line.split()[1]
            y_center = line.split()[2]
            width = line.split()[3]
            height = line.split()[4]

            if output_class < 5 and float(x_center) > 0 and float(x_center) < 1 and float(y_center) > 0 and float(y_center) < 1:
                outfile.write(f"{output_class} {x_center} {y_center} {width} {height} \n")
    outfile.close()



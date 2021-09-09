import glob

file_list = glob.glob('/home/schober/cityscape_dataset/annotations/darknet_labels_train/labels/val/*.txt')

out_folder = '/home/schober/cityscape_dataset/annotations/darknet_labels_train/labels/val_conv/'

for file in file_list:
    with open(file) as f:
        content = f.readlines()
    data_str = ''
    file_name = file.split('/')[-1]
    with open(out_folder + file_name, 'a') as outfile:
        for line in content:
            input_class = int(line.split()[0])
            if input_class == 2:
                output_class = 2
                x = line.split()[1]
                y = line.split()[2]
                w = line.split()[3]
                h = line.split()[4]
                #c = line.split()[5]
                #outfile.write(f"{output_class} {c} {x} {y} {w} {h} \n")
                outfile.write(f"{output_class} {x} {y} {w} {h} \n")
            elif input_class == 4:
                output_class = 4
                x = line.split()[1]
                y = line.split()[2]
                w = line.split()[3]
                h = line.split()[4]
                #c = line.split()[5]
                #outfile.write(f"{output_class} {c} {x} {y} {w} {h} \n")
                outfile.write(f"{output_class} {x} {y} {w} {h} \n")
            #c = line.split()[5]
            #outfile.write(f"{output_class} {c} {x} {y} {w} {h} \n")


    outfile.close()

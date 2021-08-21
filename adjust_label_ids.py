import glob

labels = sorted(glob.glob('/home/schober/cityscape_dataset/annotations/darknet_labels/labels/val/'+ '*.txt'))

for file in labels:
    i = 0
    with open(file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            cls = int(line.split()[0])
            rest = line.split()[1:]
            if cls == 1:
                cls = '0'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 5:
                cls = '1'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 2:
                cls = '2'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 3:
                cls = '3'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 0:
                cls = '4'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 0:
                cls = '5'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 9:
                cls = '6'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 6:
                cls = '8'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            if cls == 7:
                cls = '9'
                outline = ' '.join(rest)
                lines[i] = cls + ' '+outline+'\n'
            i+=1

    with open(file, 'w') as k:
        k.writelines(lines)
        k.close()

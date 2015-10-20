# -*-coding:utf-8-*- 
__author__ = 'Administrator' 
# -*- coding: UTF-8 -*-
__author__ = 'Administrator'

import csv
import __future__

from progressbar import *
loop = True
while loop:
    in_csv = raw_input('\n Please input the CSV file: \n')
    if in_csv[-4:] != '.csv':
        print("Please input a *.csv file!!!")
    in_conf = raw_input('\n Please input the point cloud file: \n')
    if in_conf[-15:] == '.xyz_label_conf':
        loop = False
    else:
        print("Please input a *.xyz_label_conf file!!!")
file = open(in_conf)
x_list = []
y_list = []
z_list = []
class_list = []
for line in file:
    if line[0] == "#":
        continue
    else:
        data = line.split(' ')
        x_list.append(float(data[0]))
        y_list.append(float(data[1]))
        z_list.append(float(data[2]))
        class_list.append(int(data[3]))
new_class_list = []
with open(in_csv) as infile:
    reader = csv.reader(infile)
    row0 = reader.next()
    for line in reader:
        if len(line) > 0:
            new_class_list.append(line[2][-4:])
outfile = open(in_conf[:-15] + '.xyz', mode='w')
count = 0
length = len(class_list)
widgets = ['Extracting features: ', Percentage(), ' ', Bar(marker=RotatingMarker('>-=')), ' ', Timer(), ' ']
p = ProgressBar(widgets=widgets).start()
while count < length:
    outfile.write(str(x_list[count]) + ' ' + str(y_list[count]) + ' ' + str(z_list[count]) + ' ' + str(class_list[count]) + ' ' + new_class_list[count] + '\n')
    p.update(int((count / (length - 1.0)) * 100))
    count += 1
p.finish()
outfile.close()

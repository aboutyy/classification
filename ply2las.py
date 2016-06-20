# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import laspy


txtpath = 'E:\Z2.txt'
laspath = txtpath[:-4] + '.las'
header = laspy.header.Header()
lasfile = laspy.file.File(laspath, header=header, mode='w')
x_array, y_array, z_array, i_array, rn_array = [], [], [], [], []
with open(txtpath, 'r') as xyzfile:
    lines = xyzfile.readlines()
    for line in lines:
        line = line.strip()
        if not len(line) or line.startswith('#'):       # 判断是否是空行或注释行
            continue
        items = line.split(',')
        x_array.append(float(items[0]))
        y_array.append(float(items[1]))
        z_array.append(float(items[2]))
        i_array.append(float(items[3]))
        rn_array.append(int(float(items[4])))
lasfile.X = x_array
lasfile.Y = y_array
lasfile.Z = z_array
min_intensity = min(i_array)
i_array = [i - min_intensity for i in i_array]
lasfile.intensity = i_array
lasfile.return_num = rn_array
lasfile.close()

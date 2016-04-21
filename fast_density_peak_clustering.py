# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import laspy
import numpy as np
import scipy.spatial
import csv
import os


def add_dimension(infile_path, outfile_path, names, types, descriptions):
    """
    add new dimensions to the las file

    Args:
        names: names array of the dimensions
        types: types array of the dimensions
                0       Raw Extra Bytes Value of “options”
                1       unsigned char   1 byte
                2       Char    1 byte
                3       unsigned short  2 bytes
                4       Short   2 bytes
                5       unsigned long   4 bytes
                6       Long    4 bytes
                7       unsigned long long      8 bytes
                8       long long       8 bytes
                9       Float   4 bytes
                10      Double  8 bytes
                11      unsigned char[2]        2 byte
                12      char[2] 2 byte
                13      unsigned short[2]       4 bytes
                14      short[2]        4 bytes
                15      unsigned long[2]        8 bytes
                16      long[2] 8 bytes
                17      unsigned long long[2]   16 bytes
                18      long long[2]    16 bytes
                19      float[2]        8 bytes
                20      double[2]       16 bytes
                21      unsigned char[3]        3 byte
                22      char[3] 3 byte
                23      unsigned short[3]       6 bytes
                24      short[3]        6 bytes
                25      unsigned long[3]        12 bytes
                26      long[3] 12 bytes
                27      unsigned long long[3]   24 bytes
                28      long long[3]    24 bytes
                29      float[3]        12 bytes
                30      double[3]       24 bytes
        description: discription of the dimension
    Returns:
        None
    """
    infile = laspy.file.File(infile_path, mode="r")
    outfile = laspy.file.File(outfile_path, mode="w", header=infile.header)
    exist_names = []
    for dimension in infile.point_format:
        exist_names.append(dimension.name)
    for name, datatype, description in zip(names, types, descriptions):
        if exist_names.count(name) != 0:
            print "dimension %s already exist!!", name
            continue
        outfile.define_new_dimension(name, datatype, description)
    for dimension in infile.point_format:
        data = infile.reader.get_dimension(dimension.name)
        outfile.writer.set_dimension(dimension.name, data)
        exist_names.append(dimension.name)
    infile.close()
    outfile.close()


def voxelization(infile_path, outfile_path, voxel_size):
    """
    voxelization of point cloud, save the voxel-first point as a file and the point-index to las file

    Args:
        infile_path: the point cloud file *.las
        outfile_path: the ASCII file that save the voxel-first point pair values, line number is voxel index and value
            is point index
        voxel_size: voxel size for voxelization

    Returns:
        None
    """

    infile = laspy.file.File(infile_path, mode="rw")
    # 计算每个点的voxel码
    scaled_x = np.vectorize(int)((1 / voxel_size) * (infile.x - infile.header.min[0]))
    scaled_y = np.vectorize(int)((1 / voxel_size) * (infile.y - infile.header.min[1]))
    scaled_z = np.vectorize(int)((1 / voxel_size) * (infile.z - infile.header.min[2]))
    indices = np.lexsort((scaled_z, scaled_y, scaled_x))
    voxel_count = 0
    point_count = 0
    point_lengh = len(infile.x)
    # the array to store the point number in each voxel
    points_in_one_voxel_array = []
    # the array to store the average intensity of points in a voxel
    intensity_in_one_voxel_array = []
    coordinate_list = []
    while point_count < point_lengh:
        # the counter of points number in one voxel
        points_in_one_voxel_count = 1
        intensity_in_one_voxel_count = infile.intensity[indices[point_count]]
        infile.gps_time[indices[point_count]] = voxel_count
        coordinate_list.append([scaled_x[indices[point_count]], scaled_y[indices[point_count]], scaled_z[indices[point_count]]])
        point_count += 1
        # loop of finding points with same code
        while point_count < point_lengh and \
                        scaled_x[indices[point_count]] == scaled_x[indices[point_count - 1]] and \
                        scaled_y[indices[point_count]] == scaled_y[indices[point_count - 1]] and \
                        scaled_z[indices[point_count]] == scaled_z[indices[point_count - 1]]:
            # add a voxel index label to the point
            infile.gps_time[indices[point_count]] = voxel_count
            intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
            point_count += 1
            points_in_one_voxel_count += 1
        intensity_in_one_voxel_array.append(intensity_in_one_voxel_count / points_in_one_voxel_count)
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        voxel_count += 1

    # save the code to the txt file sequentially
    with open(outfile_path, 'w') as txtfile:
        for coor, count, intensity in zip(coordinate_list, points_in_one_voxel_array, intensity_in_one_voxel_array):
            line = str(coor[0]) + ' ' + str(coor[1]) + ' ' + str(coor[2]) + ' ' + str(count) + ' ' + str(intensity)\
                   + '\n'
            txtfile.write(line)


def getdistance(pt1, pt2):
    tmp = pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2) + pow(pt1[2] - pt2[2], 2)
    return pow(tmp, 0.5)


def set_rho_delta(dataset, point_count_list):
    """
    密度计算函数

    此处的密度计算公式是 rho(p) = h - n / h + c/100, 其中h为当前点的z方向点累计体素数，n为当前点z方向的计数，c为这一位置点数
    比如(3,2,5)这个点，(3,2)位置有10个体素，80个点，则rho(p) = 10 - 5/10 + 0.8 = 10.3
    """
    tree = scipy.spatial.cKDTree(dataset)
    voxel_length = len(dataset)
    # 记录每个体素的密度
    rho_list = [-1] * voxel_length
    # 记录每个体素到更高密度体素的最小距离
    delta_list = [-1] * voxel_length
    # 记录一定范围内每个体素密度高于当前体素的最近体素编号
    nearest_neighbor_list = [-1] * voxel_length
    count = 0
    # 计算出所有种子
    while count < voxel_length:  # 每次循环到下一个水平位置点
        temp_seeds = [count]  # 存储的是一个水平位置点的所有体素
        # 垂直方向点计数
        point_count = point_count_list[count]
        count += 1
        # 垂直高度计数
        h_count = 1
        # 计算某一水平位置的体素集合
        if count < voxel_length:
            while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]:
                h_count += 1
                point_count += point_count_list[count]
                temp_seeds.append(count)
                count += 1
                if count >= voxel_length:
                    break
        temp_count = 0
        for seed in temp_seeds:
            # 密度函数
            rho_list[seed] = ((h_count - float(temp_count) / h_count) + point_count / 10000.0) / (dataset[:, 2][seed] + 1)
            delta = 0
            if temp_count == 0:
                temp_count += 1
                continue
            # 与密度较高体素的距离
            delta = dataset[:, 2][seed] - dataset[:, 2][seed - 1]
            if delta > 1:
                temp_count += 1
                continue
            nearest_neighbor_list[seed] = seed - 1
            delta_list[seed] = delta
            temp_count += 1

    for ii in range(voxel_length):
        if delta_list[ii] != -1:
            continue
        neighbors = tree.query_ball_point(dataset[ii], 10)
        distance_list = []
        neighbor_list = []
        for neighbor in neighbors:
            # 计算比当前点密度大到该点的距离
            if rho_list[neighbor] > rho_list[ii]:
                distance = getdistance(dataset[ii], dataset[neighbor])
                distance_list.append(distance)
                neighbor_list.append(neighbor)
        if len(distance_list) == 0:
            delta_list[ii] = 20
        else:
            delta_list[ii] = min(distance_list)
            # 找出最近且比当前点密度高的点
            nearest_neighbor_list[ii] = neighbor_list[distance_list.index(min(distance_list))]

    class_list = [-1] * voxel_length
    rho_list1 = [(rho_list[i], i) for i in range(len(rho_list))]
    rho_sorted = sorted(rho_list1, reverse=1)
    class_num = 1
    for ii in range(len(rho_sorted)):
        id_p = rho_sorted[ii][1]
        if delta_list[id_p] > 4 and rho_list[id_p] > 2:
            class_list[id_p] = class_num
            class_num += 1
        else:
            if class_list[nearest_neighbor_list[id_p]] != -1:
                class_list[id_p] = class_list[nearest_neighbor_list[id_p]]
            else:
                class_list[id_p] = 0  # 异常点的类别设为0，如rho较小但是delta比较大的点
    # import pylab as pl
    # fig1 = pl.figure(1)
    # pl.subplot(121)
    # draw_decision_graph(pl, rho_list, delta_list, class_list, class_num)
    # pl.show()
    return nearest_neighbor_list, rho_list, delta_list, class_list


def draw_decision_graph(pl, rho, delta, cl, color_num):
    cm = pl.get_cmap("RdYlGn")
    for i in range(len(rho)):
        pl.plot(rho[i], delta[i], 'o', color=cm(cl[i] * 1.0 / color_num))
    pl.xlabel(r'$\rho$')
    pl.ylabel(r'$\delta$')


def draw_decision_graph(pl, rho, delta, cl, color_num):
    cm = pl.get_cmap("RdYlGn")
    for i in range(len(rho)):
        pl.plot(rho[i], delta[i], 'o', color=cm(cl[i] * 1.0 / color_num))
    pl.xlabel(r'$\rho$')
    pl.ylabel(r'$\delta$')


if __name__ == '__main__':
    infilepath = 'data/2.las'
    outtxt = infilepath[:-4] + '.txt'
    outtxt1 = infilepath[:-4] + '1' + '.txt'
    if not os.path.exists(outtxt):
        print "\n1. voxelizing..."
        voxelization(infilepath, outtxt, 0.5)
    points_count_array, intensity_array = [], []
    original_x_int_array, original_y_int_array, original_z_int_array = [], [], []
    with open(outtxt, 'r') as txt_file:
        fin = txt_file.readlines()
        for line in fin:
            line = line.strip()
            if not len(line) or line.startswith('#'):       # 判断是否是空行或注释行
                continue
            items = line.split(' ')
            original_x_int_array.append(int(items[0]))
            original_y_int_array.append(int(items[1]))
            original_z_int_array.append(int(items[2]))
            points_count_array.append(int(items[3]))
            intensity_array.append(int(items[4]))

    original_dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    print "\n2. clustering..."
    neighbors, rhos, deltas, classes = set_rho_delta(original_dataset, points_count_array)
    with open(outtxt, 'r') as fin, open(outtxt1, 'w') as fout:
        lines = fin.readlines()
        for line, neighbor, rhoitem, deltaitem, classitem in zip(lines, neighbors, rhos, deltas, classes):
            line = line.strip()
            line = line + ' ' + str(neighbor) + ' ' + str(rhoitem) + ' ' + str(deltaitem) + ' ' + str(classitem) + '\n'
            fout.write(line)
    lasfile = laspy.file.File(infilepath, mode="rw")
    point_length = len(lasfile.x)
    point_count = 0
    lasfile.raw_classification[:] = 0
    lasfile.pt_src_id[:] = 0
    for time in lasfile.gps_time:
        lasfile.pt_src_id[point_count] = classes[int(time)]
        point_count += 1
    lasfile.close()

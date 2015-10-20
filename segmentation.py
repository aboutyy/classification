# -*- coding: utf-8 -*-
# Created by You Li on 2015-03-09 0009
import laspy
import timeit
import os
import csv
import numpy as np
import scipy
from scipy import linalg as la
import scipy.spatial

# ####### 定义常量###### #
# 体素大小
VOXEL_SIZE = 0.15

# 最小杆高度
MIN_HEIGHT = 0.9

# 杆离地面最大距离
DISTANCE_TO_GROUND = 1.0

# 杆在地面的最大面积
MAX_AREA = 2

# 邻居最远体素距离
MAX_NEIGHBOR_DISTANCE = 0.3

# 判断地面的法向量
GROUND_NORMAL_THRESHOLD = 0.7

# 作圆柱判断的最小圆柱高度
MIN_CYLINDER_HEIGHT = 1.0

# 内圆柱半径
INNER_RADIUS = 0.3

MAX_INNER_RADIUS = 0.7
# 双圆柱内外圆柱之间的距离
DISTANCE_OF_IN2OUT = 0.3

# 双圆柱用来定义杆的内外点比例
RATIO_OF_POINTS_COUNT = 0.98

# 是否进行地面距离判定
USE_GROUND = True

# 地面点判断时的邻居点最远距离
GROUND_NEIGHBOR = 0.5

# 合并相邻垂直voxel组的距离阈值
MERGING_DISTANCE = 0.3

# 过滤长宽
FILTERING_LENGTH = 1.0


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

    # the array to store the code of the voxel, this is actually the row, columm and height number of the voxel
    code_array = []

    # the array to store the point number in each voxel
    points_in_one_voxel_array = []

    # the array to store the average intensity of points in a voxel
    intensity_in_one_voxel_array = []

    while point_count < point_lengh - 1:

        # the counter of points number in one voxel
        points_in_one_voxel_count = 1
        intensity_in_one_voxel_count = 0
        # loop of finding points with same code
        while point_count < point_lengh - 1 and \
                        scaled_x[indices[point_count + 1]] == scaled_x[indices[point_count]] and \
                        scaled_y[indices[point_count + 1]] == scaled_y[indices[point_count]] and \
                        scaled_z[indices[point_count + 1]] == scaled_z[indices[point_count]]:
            # add a voxel index label to the point
            infile.voxel_index[indices[point_count]] = voxel_count
            intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
            point_count += 1
            points_in_one_voxel_count += 1

        infile.voxel_index[indices[point_count]] = voxel_count
        intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
        intensity_in_one_voxel_array.append(intensity_in_one_voxel_count / points_in_one_voxel_count)
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        code = "{:0>4d}".format(scaled_x[indices[point_count]]) + \
               "{:0>4d}".format(scaled_y[indices[point_count]]) + \
               "{:0>4d}".format(scaled_z[indices[point_count]])
        code_array.append(code)
        point_count += 1
        voxel_count += 1

    # save the code to the csv file sequentially
    code_array_length = len(code_array)
    with open(outfile_path, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        while count < code_array_length:
            writer.writerow([code_array[count], points_in_one_voxel_array[count], intensity_in_one_voxel_array[count]])
            count += 1


def vertical_continiuity_analysis(dataset, point_count_array):
    """
    向上连续性分析，分析出具有连续性的位置点

    通过从最低位置点开始，向上面方向的邻居做增长，选取包含最多点的体素作为增长的方向，依次类推，直到没有了向上的体素为止。
    """
    # 存储所有位置的最低点作为增长的种子点
    seeds_list = []
    seeds_list.append(0)
    voxel_length = len(dataset)
    count = 1
    flag = 0
    # 计算出所有种子
    while count < voxel_length:
        # 与前一个体素x,y 不相同，找的是最低的体素
        temp_seeds = []
        while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]:
            temp_seeds.append(count)
            count += 1
            if count >= voxel_length:
                break
        if len(temp_seeds) != 0:
            # 找出向上连续增长少于一定数目的点当作地面点
            temp_temp_seeds = []
            for temp_seed in temp_seeds:
                if dataset[:, 2][temp_seed] - dataset[:, 2][temp_seed - 1] < 3:
                    temp_temp_seeds.append(temp_seed)
                else:
                    break
                if len(temp_temp_seeds) >= 3:
                    break
            if len(temp_temp_seeds) < 3:
                seeds_list += temp_temp_seeds
        else:
            seeds_list.append(count)
            count += 1
    return seeds_list


if __name__ == '__main__':
    import yylog

    loop = True
    while loop:
        inputpath = raw_input('\n Input las file name: \n')
        infilepath = inputpath + '.las'
        if os.path.exists(infilepath):
            loop = False
        else:
            print 'File not exist!!'
            loop = True
    outlas = inputpath + '_' + str(VOXEL_SIZE)  + '.las'
    outcsv = outlas[:-4] + '.csv'
    outcsv1 = outcsv[:-4] + '_1' + '.csv'

    # ###################新建加入新字段的las文件###################
    # 如果已经添加过字段了就不用再添加
    if not os.path.exists(outlas):
        print "Adding dimensions..."
        add_dimension(infilepath, outlas, ["voxel_index", "tlocation", "olocation", "flocation"], [5, 5, 5, 5],
                      ["voxel num the point in", "original location label", "merged location label", "temp"])

    # ############## 1.体素化 ################
    # 如果体素化了下一次就不用体素化了
    if not os.path.exists(outlas) or not os.path.exists(outcsv):
        print "\n1. voxelizing..."
        voxelization(outlas, outcsv, VOXEL_SIZE)

    with open(outcsv, 'rb') as in_csv_file:
        reader = csv.reader(in_csv_file)
        line = [[row[0], row[1], row[2]] for row in reader]
    voxel_code_array = np.array(line)[:, 0]
    points_count_array = np.vectorize(int)(np.array(line)[:, 1])
    intensity_array = np.vectorize(int)(np.array(line)[:, 2])
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], voxel_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], voxel_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], voxel_code_array))
    original_dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()

    # ############# 2.垂直连续性分析 ################
    log = yylog.LOG('pole')
    # try:
    start = timeit.default_timer()
    print '\n2. vertical_continiuity_analysis...'
    ground_voxels = vertical_continiuity_analysis(original_dataset, points_count_array)

    print '\n6. lableling...'
    ground_voxels_array = np.array([0] * len(voxel_code_array))
    count = 1
    for ground_voxel in ground_voxels:
        ground_voxels_array[ground_voxel] = 1

    lasfile = laspy.file.File(outlas, mode="rw")
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    if USE_GROUND:
        lasfile.raw_classification[:] = 0
    lasfile.pt_src_id[:] = 0
    for voxel_index in lasfile.voxel_index:
        # lasfile.gps_time[point_count] = horizontal_location_array[voxel_index]
        lasfile.raw_classification[point_count] = ground_voxels_array[voxel_index]
        point_count += 1
    lasfile.close()

    # except:
    #     log.error()  # 使用系统自己的错误描述
    #     os.system('pause')
    #     exit()
    os.system('pause')

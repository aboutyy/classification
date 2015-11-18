# -*- coding: utf-8 -*-
# Created by You Li on 2015-03-09 0009
import laspy
import timeit
import os
import math
import csv
import numpy as np
import scipy
import geometry
import progressbar
import scipy.spatial

# ####### 定义常量###### #
# 体素大小
VOXEL_SIZE = 0.15

# 判断地面的法向量
GROUND_NORMAL_THRESHOLD = 0.7

# 是否进行地面距离判定
USE_GROUND = True

# 地面局部偏低的计算范围
GROUND_LOCAL_DISTANDE = 2.0

# 地面最大角度
MAX_SLOPE = 15


def preprocess():
    """
    点云体素的预处理

    通过预处理去除一些特别稀疏的体素，为后面的计算去除一些问题，比如：计算法向量时计算矩阵是需要至少三个点
    """
    pass


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


def ground_detection(dataset, point_count_array):
    """
    地面点的提取

    通过从最低位置点开始，向上增长，如果没有向上点或者向上点不够，则作为潜在地面点，再通过其他的限制条件比如法向量，维度等条件再筛选
    """
    #
    widgets = ['ground_detection: ', progressbar.Percentage(), ' ', progressbar.Bar(),
               ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=100).start()
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
    pbar.update(19)
    tree = scipy.spatial.cKDTree(dataset)
    updated_seeds = []
    count = 0
    for seed in seeds_list:
        normal = geometry.get_normals(dataset, tree, dataset[:, 0][seed], dataset[:, 1][seed], dataset[:, 2][seed], 4)
        # 离散点不认为是地面点
        if normal < -1:
            continue
        if abs(normal) > GROUND_NORMAL_THRESHOLD:
            updated_seeds.append(seed)
        count += 1
        pbar.update(19 + 50 * float(count) / len(seeds_list))
    new_dataset = np.vstack([dataset[:, 0], dataset[:, 1]]).transpose()  # 投影三维数据到二维平面，构建二维的数据集
    new_tree = scipy.spatial.cKDTree(new_dataset)
    final_seeds = []
    # 判断是否局部处于底部区域，默认最大坡度15°。2米范围内的高差是2 * sin(15°) = 0.52。 如果该点与周围点中最低点距离大于这个值不被认为是地面点
    count = 0
    for new_seed in updated_seeds:
        neighbors = new_tree.query_ball_point(new_dataset[new_seed], GROUND_LOCAL_DISTANDE / VOXEL_SIZE)
        minz = min(dataset[:, 2][neighbors])
        if dataset[:, 2][new_seed] - minz < 2 * math.sin(3.14 * (MAX_SLOPE / 180.0)) / VOXEL_SIZE:
            final_seeds.append(new_seed)
        count += 1
        pbar.update(70 + float(count) * 30 / len(updated_seeds))
    pbar.finish()
    return final_seeds


def object_position_detection(dataset, point_count_array):
    """

    向上连续性分析，分析出具有连续性的位置点

    通过从最低位置点开始，向上面方向的邻居做增长，选取包含最多点的体素作为增长的方向，依次类推，直到没有了向上的体素为止。
    """

    # 存储所有位置的最低点作为增长的种子点
    seeds_list = []
    voxel_length = len(dataset)
    count = 1
    previous_x = dataset[:, 0][0]
    previous_y = dataset[:, 1][0]
    flag = 0
    # 计算出所有种子
    while count < voxel_length:
        if dataset[:, 0][count] == previous_x and dataset[:, 1][count] == previous_y:
            if dataset[:, 2][count] - dataset[:,2][count-1] < 3 and flag == count - 1:
                # 过滤边缘点
                #if points_count_array[count] > 1:
                seeds_list.append(count)
        else:
            flag = count
            previous_x = dataset[:, 0][count]
            previous_y = dataset[:, 1][count]
        count += 1
    tree = scipy.spatial.cKDTree(dataset)
    # 存储3维位置点信息
    location_list_list = []
    # 存储水平位置点集合
    horizontal_location_list = []
    count = 0
    for seed in seeds_list:
        count += 1
        location_list = []
        vertical_count = 0
        current_seed = seed
        location_list.append(current_seed)
        # 选择26邻居体素
        while True:
            neighbors = tree.query_ball_point(dataset[current_seed], float(MAX_NEIGHBOR_DISTANCE) / VOXEL_SIZE)
            neighbors = np.array(neighbors)
            if len(neighbors) <= 1:
                break
            else:
                # 找出上邻居点
                up_indexs = np.where(dataset[:, 2][neighbors] - dataset[:, 2][current_seed] == 1)[0]
                # 找出正上点
                up_index = np.where((dataset[:, 2][neighbors] - dataset[:, 2][current_seed] == 1) &
                                    (dataset[:, 0][neighbors] == dataset[:, 0][current_seed]) &
                                    (dataset[:, 1][neighbors] == dataset[:, 1][current_seed]))[0]
                up_neighbor_lenght = len(up_indexs)
                if up_neighbor_lenght > 0:
                    vertical_count += 1
                    if up_neighbor_lenght == 1:
                        current_seed = neighbors[up_indexs][0]
                    elif len(up_index) != 0:
                        current_seed = neighbors[up_index[0]]
                    else:
                        temp_index = np.where(point_count_array[neighbors[up_indexs]] ==
                                              max(point_count_array[neighbors[up_indexs]]))[0][0]
                        current_seed = neighbors[up_indexs[temp_index]]
                    # 加入所有邻居点到潜在杆位置点中
                    for index in neighbors[up_indexs]:
                        location_list.append(index)
                else:
                    break
        # 若向上增长能达到一定高度，则被认为是一个潜在的位置点
        height = max(dataset[:, 2][location_list]) - min(dataset[:, 2][location_list])
        if height * VOXEL_SIZE >= MIN_HEIGHT:
            location_list_list.append(location_list)
            horizontal_location_list.append(seed)
    return horizontal_location_list, location_list_list


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
    print '\n2. Ground detecting...'
    ground_voxels = ground_detection(original_dataset, points_count_array)

    print '\n3. lableling...'
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

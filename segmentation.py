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
from sklearn.cluster import DBSCAN
from sklearn import linear_model, datasets
# ####### 定义常量###### #
# 体素大小
VOXEL_SIZE = 0.2

# 判断地面的法向量
GROUND_NORMAL_THRESHOLD = 0.7

# 是否进行地面距离判定
USE_GROUND = True

# 地面局部偏低的计算范围
GROUND_LOCAL_DISTANDE = 5.0

# 地面最大角度
MAX_SLOPE = 15

# 地面点最大高度
GROUND_HEIGHT = 0.6

# 最小位置高度，大于等于这个高度则认为是一个地物点
POSITION_HEIGHT = 0.8

# 位置点离地面最大距离
DISTANCE_TO_GROUND = 1.0


def preprocess():
    """
    点云体素的预处理

    通过预处理去除一些特别稀疏的体素，为后面的计算去除一些问题，比如：计算法向量时计算矩阵是需要至少三个点
    """
    pass


def region_growing(voxelset, radius, angle_threshold):
    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation
    # Point cloud: voxelset
    # Point Normals: normal_list
    # Angle threshold: angle_threshold
    # Awailable point list: a_list

    tree = scipy.spatial.cKDTree(voxelset)
    length = len(voxelset)
    a_list = range(length)
    seed_length = len(a_list)
    # Point normals
    normal_list = []
    for voxel in voxelset:
        normal_list.append(math.fabs(geometry.get_normal(voxelset, tree, voxel[0], voxel[1], voxel[2], 2)))
    # region list
    regions = []
    while len(a_list) > 0:
        current_region = []
        current_seeds = []
        # voxel with lowest z value
        lowest_voxel_indice = a_list[0]
        current_seeds.append(lowest_voxel_indice)
        current_region.append(lowest_voxel_indice)
        del a_list[0]
        count = 0
        while count < len(current_seeds):
            current_seed = current_seeds[count]
            count += 1
            current_seed_neighbors = tree.query_ball_point([voxelset[:, 0][current_seed], voxelset[:, 1][current_seed],
                                                            voxelset[:, 2][current_seed]], radius)
            for neighbor in current_seed_neighbors:
                if a_list.count(neighbor) != 0:
                    if current_region.count(neighbor) == 0 and math.acos(math.fabs(normal_list[neighbor] - normal_list[current_seed])) < angle_threshold:
                        current_region.append(neighbor)
                        a_list.remove(neighbor)
                        if current_seeds.count(neighbor) == 0:
                            current_seeds.append(neighbor)
        regions.append(np.array(current_region))
    return regions


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
    while point_count < point_lengh:
        # the counter of points number in one voxel
        points_in_one_voxel_count = 1
        intensity_in_one_voxel_count = infile.intensity[indices[point_count]]
        infile.voxel_index[indices[point_count]] = voxel_count
        code = "{:0>4d}".format(scaled_x[indices[point_count]]) + \
       "{:0>4d}".format(scaled_y[indices[point_count]]) + \
       "{:0>4d}".format(scaled_z[indices[point_count]])
        code_array.append(code)
        point_count += 1
        # loop of finding points with same code
        while point_count < point_lengh and \
                        scaled_x[indices[point_count]] == scaled_x[indices[point_count - 1]] and \
                        scaled_y[indices[point_count]] == scaled_y[indices[point_count - 1]] and \
                        scaled_z[indices[point_count]] == scaled_z[indices[point_count - 1]]:
            # add a voxel index label to the point
            infile.voxel_index[indices[point_count]] = voxel_count
            intensity_in_one_voxel_count += infile.intensity[indices[point_count]]
            point_count += 1
            points_in_one_voxel_count += 1
        intensity_in_one_voxel_array.append(intensity_in_one_voxel_count / points_in_one_voxel_count)
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        voxel_count += 1

    # save the code to the csv file sequentially
    code_array_length = len(code_array)
    with open(outfile_path, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        while count < code_array_length:
            writer.writerow([code_array[count], points_in_one_voxel_array[count], intensity_in_one_voxel_array[count]])
            count += 1


def ground_and_position_detection(dataset, point_count_array):
    """
    地面点的提取

    通过从最低位置点开始，向上增长，如果没有向上点或者向上点不够，则作为潜在地面点，再通过其他的限制条件比如法向量，维度等条件再筛选
    """
    widgets = ['ground_detection: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='>'), ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=100).start()
    seeds_list = []
    voxel_length = len(dataset)
    count = 0
    # 记录每个水平位置点对应的垂直位置
    position_list_list = []
    # 计算出所有种子
    while count < voxel_length - 1:  # 每次循环到下一个水平位置点
        pbar.update(19 * float(count) / voxel_length)
        temp_seeds = [count]  # 存储的是一个水平位置点的所有体素
        count += 1
        # 与前一个体素x,y 不相同，找的是最低高度的体素
        while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]:
            temp_seeds.append(count)
            count += 1
            if count >= voxel_length:
                break
        if len(temp_seeds) > 1:
            # 找出向上连续增长少于一定数目的点当作地面点
            temp_temp_seeds = [temp_seeds[0]]  # 存储的是在垂直方向上连续增长的体素
            for index in range(0, len(temp_seeds) - 1):
                # 判断连续性的条件是垂直方向上间隔不超过1个体素
                if dataset[:, 2][temp_seeds[index + 1]] - dataset[:, 2][temp_seeds[index]] < 4:
                    temp_temp_seeds.append(temp_seeds[index + 1])
                else:
                    break
            # 垂直方向上小于一定高度则认为是地面点
            if len(temp_temp_seeds) <= GROUND_HEIGHT / VOXEL_SIZE:
                seeds_list += temp_temp_seeds
            # 垂直方向大于一定高度则认为是地物点
            elif len(temp_temp_seeds) >= POSITION_HEIGHT / VOXEL_SIZE:
                position_list_list.append(temp_temp_seeds)
        else:
            seeds_list.append(count - 1)
    if dataset[:, 0][voxel_length - 1] != dataset[:, 0][voxel_length - 2] or dataset[:, 1][voxel_length - 1] != dataset[:, 1][voxel_length - 2]:
        seeds_list.append(voxel_length - 1)
    pbar.update(19)
    tree = scipy.spatial.cKDTree(dataset)
    updated_seeds = []
    count = 0
    for seed in seeds_list:
        normal = geometry.get_normal(dataset, tree, dataset[:, 0][seed], dataset[:, 1][seed], dataset[:, 2][seed], 4)
        # 离散点不认为是地面点
        if normal < -1:
            continue
        if abs(normal) > GROUND_NORMAL_THRESHOLD:
            updated_seeds.append(seed)
        count += 1
        pbar.update(19 + 50 * count / len(seeds_list))
    new_dataset = np.vstack([dataset[:, 0], dataset[:, 1]]).transpose()  # 投影三维数据到二维平面，构建二维的数据集
    new_tree = scipy.spatial.cKDTree(new_dataset)
    final_ground_seeds = []
    # ##判断是否局部处于底部区域1.默认最大坡度15°。2米范围内的高差是2 * sin(15°) = 0.52。如果该点与周围点中最低点距离大于这个值不被认为是地面点
    # ##2. 如果该点与平均值相差太远也不属于地面点
    count = 0
    for new_seed in updated_seeds:
        neighbors = new_tree.query_ball_point(new_dataset[new_seed], GROUND_LOCAL_DISTANDE / VOXEL_SIZE)
        minz = min(dataset[:, 2][neighbors])
        if dataset[:, 2][new_seed] - minz < 5 * math.sin(3.14 * (MAX_SLOPE / 180.0)) / VOXEL_SIZE:
            final_ground_seeds.append(new_seed)
        count += 1
        pbar.update(70 + count * 30 / len(updated_seeds))
    # ##判断位置点与地面点的距离，以排除一些非位置点
    filtered_position_list_list = []
    back_dataset = dataset[final_ground_seeds]
    back_tree = scipy.spatial.cKDTree(back_dataset)
    # for position in position_list_list:
    #     distance, neighbor = back_tree.query(dataset[position[0]])  # 位置点里面的第一个点是最低点，计算最点与地面的距离
    #     z_neighbor = dataset[:, 2][final_ground_seeds[neighbor]]
    #     if dataset[:, 2][position[0]] - z_neighbor < DISTANCE_TO_GROUND / VOXEL_SIZE:
    #         filtered_position_list_list.append(position)
    pbar.finish()
    indices = []
    h_indices = []
    for position in position_list_list:
        indices += position
        h_indices.append(position[0])
    # Compute DBSCAN
    X = dataset[indices]
    h_X = dataset[:, 0:2][h_indices]
    db = DBSCAN(eps=3, min_samples=2).fit(X)
    h_db = DBSCAN(eps=3, min_samples=3).fit(h_X)
    location_array = np.array([0] * len(point_count_array))
    h_location_array = np.array([0] * len(point_count_array))
    location_array[indices] = db.labels_ + 1
    h_location_array[h_indices] = h_db.labels_ + 1
    return final_ground_seeds, location_array, h_location_array


def feature_extraction(dataset, point_count_array):
    """

    对位置点块提取特征，为下一步分类做好准备

    """

    return


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
    voxel_length = len(voxel_code_array)

    # ############# 2.垂直连续性分析 ################
    log = yylog.LOG('pole')
    # try:
    start = timeit.default_timer()
    print '\n2. Ground and Positions detecting...'
    ground_voxels, position_list_array, horizontal_list_array = ground_and_position_detection(original_dataset, points_count_array)
    horizontal_list_list = []
    horizontal_number_list = []
    for position in horizontal_list_array:
        if position in horizontal_number_list:
            index = horizontal_number_list.index(position)
            horizontal_list_array[index].append(position)
        else:
            temp_list = [position]
            horizontal_list_list.append(temp_list)
    for horizontal_list in horizontal_list_list:
        if len(horizontal_list) < 5:
            continue
        X = original_x_int_array[horizontal_list]
        y = original_y_int_array[horizontal_list]
        # Robustly fit linear model with RANSAC algorithm
        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
        model_ransac.fit(X, y)
        inlier_mask = model_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

    print '\n3. Lableling...'
    ground_voxels_array = np.array([0] * len(voxel_code_array))
    count = 1
    for ground_voxel in ground_voxels:
        ground_voxels_array[ground_voxel] = 1
    lasfile = laspy.file.File(outlas, mode="rw")
    point_length = len(lasfile.x)
    point_count = 0
    lasfile.user_data[:] = 0
    lasfile.gps_time[:] = 0
    lasfile.olocation[:] = 0
    if USE_GROUND:
        lasfile.raw_classification[:] = 0
    lasfile.pt_src_id[:] = 0
    widgets = ['ground_detection: ', progressbar.Percentage(), ' ', progressbar.Bar(),
               ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=10000).start()
    for voxel_index in lasfile.voxel_index:
        # lasfile.gps_time[point_count] = horizontal_location_array[voxel_index]
        lasfile.raw_classification[point_count] = ground_voxels_array[voxel_index]
        lasfile.olocation[point_count] = position_list_array[voxel_index]
        lasfile.gps_time[point_count] = horizontal_list_array[voxel_index]
        point_count += 1
        pbar.update((point_count - 1) * 10000 / point_length)
    lasfile.close()
    pbar.finish()
    # except:
    #     log.error()  # 使用系统自己的错误描述
    #     os.system('pause')
    #     exit()
    os.system('pause')

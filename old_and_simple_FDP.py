# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import laspy
import numpy as np
import scipy.spatial
import math
import os
import progressbar
from tqdm import tqdm, trange

# 一些基本参数：物体最大半径5米，聚类中心点与其他高密度中心点距离阈值为3米,地面点最高高度为0.5米

# TODO 测试较小的体素大小
VOXEL_SIZE = 0.4
# 地面点最大高度，超过则不为地面点
GROUND_HEIGHT = 0.5 / VOXEL_SIZE
# 地面最大角度°
MAX_SLOPE = 15
# 地面局部偏低的计算范围，5米之内如果不是局部低点的话则不是地面点
GROUND_LOCAL_DISTANDE = 5.0
DELTA_THRESHOLD = 2 / VOXEL_SIZE
RHO_THRESHOLD = 1.5 / VOXEL_SIZE

# 这个即是规定的最大物体半径
NEIGHBOUR_DISTANCE = int(5 / VOXEL_SIZE)


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
        intensity_in_one_voxel_array.append(intensity_in_one_voxel_count / float(points_in_one_voxel_count))
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        voxel_count += 1

    # save the code to the txt file sequentially
    with open(outfile_path, 'w') as txtfile:
        for coor, count, intensity in zip(coordinate_list, points_in_one_voxel_array, intensity_in_one_voxel_array):
            line = str(coor[0]) + ' ' + str(coor[1]) + ' ' + str(coor[2]) + ' ' + str(count) + ' ' + str(intensity)\
                   + '\n'
            txtfile.write(line)


def seed_fill_connected_component_analysis_old(dataset):
    data_length = len(dataset)
    label = 1
    label_array = np.array([1] * data_length)
    print '\n seed filling...'
    for count in tqdm(range(data_length)):
        if label_array[count] == 1:
            neighbor_voxels = []
            neighbor_voxels.append(count)
            label += 1
            while len(neighbor_voxels) > 0:
                cur_voxel = dataset[neighbor_voxels[-1]]
                # label the current voxel
                label_array[neighbor_voxels[-1]] = label
                # pop the voxel
                neighbor_voxels.pop()
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if i == 0 and j == 0 and k == 0:
                                continue
                            else:
                                index = np.where((dataset[:, 0] == cur_voxel[0] + i) & (dataset[:, 1] == cur_voxel[1] + j) & (dataset[:, 2] == cur_voxel[2] + k))[0]
                                if index.size == 0:
                                    continue
                                else:
                                    if label_array[index] == 1:
                                        neighbor_voxels.append(index[0])
    return label_array


def seed_fill_connected_component_analysis(dataset):
    data_length = len(dataset)
    rows = max(dataset[:, 0]) + 2
    cols = max(dataset[:, 1]) + 2
    height = max(dataset[:, 2]) + 2
    whole_label_set = np.zeros(shape=(rows, cols, height), dtype=int)
    for count in range(data_length):
        x, y, z = dataset[:, 0][count], dataset[:, 1][count], dataset[:, 2][count]
        whole_label_set[x, y, z] = 1
    label = 1
    for count in trange(data_length, desc="Seed Filling"):
        x, y, z = dataset[:, 0][count], dataset[:, 1][count], dataset[:, 2][count]
        if whole_label_set[x, y, z] == 1:
            neighbor_voxels = []
            neighbor_voxels.append([x, y, z])
            label += 1
            while len(neighbor_voxels) > 0:
                [x1, y1, z1] = neighbor_voxels[-1]
                # label the current voxel
                whole_label_set[x1, y1, z1] = label
                # pop the voxel
                neighbor_voxels.pop()
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if i == 0 and j == 0 and k == 0:
                                continue
                            elif x1 + i < 0 or y1 + j < 0 or z1 + k < 0:
                                continue
                            else:
                                if whole_label_set[x1 + i, y1 + j, z1 + k] == 1:
                                    neighbor_voxels.append([x1 + i, y1 + j, z1 + k])
    label_array = [1] * data_length
    for m in range(data_length):
        label_array[m] = whole_label_set[dataset[:, 0][m], dataset[:, 1][m], dataset[:, 2][m]]
    return label_array


def ground_detection(dataset, point_count_array):
    """
    地面点的提取
    通过从最低位置点开始，向上增长，如果没有向上点或者向上点不够，则作为潜在地面点，再通过其他的限制条件比如法向量，维度等条件再筛选
    """
    pbar = tqdm(desc='ground detecting', total=100)
    seeds_list = []
    voxel_length = len(dataset)
    count = 0
    # 计算出所有种子
    while count < voxel_length - 1:  # 每次循环到下一个水平位置点
        pbar.update(19 * float(count) / voxel_length)
        temp_seeds = [count]  # 存储的是一个水平位置点的所有体素
        count += 1
        if count < voxel_length:
            # 与前一个体素x,y 不相同，找的是最低高度的体素
            while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]:
                if dataset[:, 2][temp_seeds[count]] - dataset[:, 2][temp_seeds[count - 1]] < 2:
                    temp_seeds.append(count)
                    count += 1
                    if count >= voxel_length:
                        break
            if len(temp_seeds) <= GROUND_HEIGHT:
                for seed in temp_seeds:
                    seeds_list.append(seed)
    pbar.update(19)
    tree = scipy.spatial.cKDTree(dataset)
    updated_seeds = []

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
    pbar.close()
    return final_ground_seeds


def get_distance(pt1, pt2):
    tmp = (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2
    return tmp ** 0.5


def get_compund_distance(pt1, pt2, i1, i2):
    tmp = pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2) + pow(pt1[2] - pt2[2], 2)
    u_distance = pow(tmp, 0.5)
    i_distance = math.fabs(i1 - i2)
    return u_distance + i_distance


def set_rho_delta(dataset, point_count_list, intensity_list, labels_list):
    """
    密度计算函数
    此处的密度计算公式是 rho(p) = h - n / h + c/100, 其中h为当前点的z方向点累计体素数，n为当前点z方向的计数，c为这一位置点数
    比如(3,2,5)这个点，(3,2)位置有10个体素，80个点，则rho(p) = 10 - 5/10 + 0.8 = 10.3
    """
    data_length = len(dataset)
    rows = max(dataset[:, 0]) + 2
    cols = max(dataset[:, 1]) + 2
    height = max(dataset[:, 2]) + 2
    whole_label_set = np.zeros(shape=(rows, cols, height), dtype=int)
    for count in range(data_length):
        x, y, z = dataset[:, 0][count], dataset[:, 1][count], dataset[:, 2][count]
        whole_label_set[x, y, z] = 1  # 有数据的格子标记为1
    new_dataset = np.vstack([dataset[:, 0], dataset[:, 1]]).transpose()
    hor_tree = scipy.spatial.cKDTree(new_dataset)
    # 记录每个体素的密度
    rho_list = [-1] * data_length
    # 记录每个体素到更高密度体素的最小距离
    delta_list = [-1] * data_length
    # 记录一定范围内每个体素密度高于当前体素的最近体素编号
    nearest_neighbor_list = [-1] * data_length
    count = 0
    intensity_range = float(max(intensity_list) - min(intensity_list))
    # 计算出所有种子
    seed_list = []
    while count < data_length:  # 每次循环到下一个连续水平位置点
        temp_seeds = [count]  # 存储的是一个水平位置点的所有体素
        # 垂直方向点计数
        count += 1
        # 垂直高度计数
        h_count = 1
        # 计算某一水平位置的连续体素集合
        if count < data_length:
            while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]\
                    and dataset[:, 2][count] - dataset[:, 2][count - 1] <= 0.5 / VOXEL_SIZE:
                h_count += 1
                temp_seeds.append(count)
                count += 1
                if count >= data_length:
                    break
        seed_list.append(temp_seeds)
    temp_list = range(-NEIGHBOUR_DISTANCE, NEIGHBOUR_DISTANCE + 1)
    for i in trange(len(seed_list), desc='computing rho'):
        temp_seeds = seed_list[i]
        temp_count = 0
        if dataset[:, 2][temp_seeds[0]] == 0:
            distance = 0
        else:
            distance_list = []
            for i in temp_list:
                for j in temp_list:
                    for k in range(- dataset[:, 2][temp_seeds[0]], 0):
                        if whole_label_set[i, j, k] == 1:
                            distance_list.append(-k)
                            break
            if len(distance_list) == 0:
                distance = 0
            else:
                distance = max(distance_list)
        h_count = len(temp_seeds)
        point_count = sum(np.array(points_count_array)[temp_seeds])
        for seed in temp_seeds:
            # 密度函数
            # todo 1.修改rho计算函数，垂直方向的高度重计算
            # rho_list[seed] = ((h_count - float(temp_count) / h_count) + point_count / 10000.0) / (dataset[:, 2][seed] + 1)
            # 如果与地面距离小于1米则直接计算，不考虑距离
            if distance <= (1 / VOXEL_SIZE):
                rho_list[seed] = h_count - float(temp_count) / h_count + point_count / 100000.0
            # 如果与地面距离大于1米则与到地面距离成反比
            else:
                rho_list[seed] = ((h_count - float(temp_count) / h_count) + point_count / 10000.0) / (1 / VOXEL_SIZE + distance)
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
    for ii in trange(data_length, desc='computing delta'):
        if delta_list[ii] != -1:
            continue
        neighbors = hor_tree.query_ball_point(new_dataset[ii], NEIGHBOUR_DISTANCE)
        distance_array = []
        neighbor_array = []
        for neighbor in neighbors:
            # 计算比当前点密度大到该点的距离
            if rho_list[neighbor] > rho_list[ii]:
                distance = get_distance(dataset[ii], dataset[neighbor])
                # TODO 把intensity加入距离计算之中
                # compund_distance = distance + 2 *  math.fabs((intensity_list[ii] - intensity_list[neighbor])) / intensity_range
                if labels_list[ii] != labels_list[neighbor]:
                    distance += NEIGHBOUR_DISTANCE
                distance_array.append(distance)
                neighbor_array.append(neighbor)
        if len(distance_array) == 0:
            delta_list[ii] = 20
        else:
            delta_list[ii] = min(distance_array)
# 当到周围的高密度点距离都是大于最大物体半径时，说明该物体周围高密度点都是与其不连接的，则该点没有有任何邻居，除非可做聚类中心点，最后会被当作聚类中心或异常点
            if delta_list[ii] < NEIGHBOUR_DISTANCE:
                nearest_neighbor_list[ii] = neighbor_array[distance_array.index(delta_list[ii])]

    class_list = [-1] * data_length
    rho_list1 = [(rho_list[i], i) for i in range(len(rho_list))]
    rho_sorted = sorted(rho_list1, reverse=1)
    class_num = 1

    for ii in trange(len(rho_sorted), desc='labeling'):
        id_p = rho_sorted[ii][1]
        if delta_list[id_p] > DELTA_THRESHOLD and rho_list[id_p] > RHO_THRESHOLD:
            class_list[id_p] = class_num
            class_num += 1
        else:
            if nearest_neighbor_list[id_p] != -1:
                class_list[id_p] = class_list[nearest_neighbor_list[id_p]]
            else:  # 规定如果没有邻居点，则当作异常点
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


if __name__ == '__main__':
    infilepath = 'data/2.las'
    outtxt = infilepath[:-4] + '_' + str(VOXEL_SIZE) + '.txt'
    if not os.path.exists(outtxt):
        print "\n1. voxelizing..."
        voxelization(infilepath, outtxt, VOXEL_SIZE)
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
            intensity_array.append(float(items[4]))

    original_dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    print "\n2. clustering..."
    labels_list = seed_fill_connected_component_analysis(original_dataset)
    neighbors, rhos, deltas, classes = set_rho_delta(original_dataset, points_count_array, intensity_array, labels_list)
    with open(outtxt, 'w') as fout:
        for x, y, z, p, i, l, neighbor, rhoitem, deltaitem, classitem in zip(original_x_int_array, original_y_int_array, original_z_int_array, points_count_array, intensity_array, labels_list, neighbors, rhos, deltas, classes):
            line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(p) + ' ' + str(i) + ' ' + str(l)
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

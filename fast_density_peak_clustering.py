# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import laspy
import numpy as np
import scipy.spatial
import math
import os
import progressbar

# 判断是否连续的阈值
CONNECTIVITY_THRESHOLD = 0.6
# 两个物体聚类中心点之间的最小距离
OBJ_DISTANCE = 2
# TODO 测试较小的体素大小
VOXEL_SIZE = 0.5
# 地面点最大高度
GROUND_HEIGHT = 0.5
# 地面最大角度
MAX_SLOPE = 10
# 地面局部偏低的计算范围,如5米之内最高距离差不应该大于0.87
GROUND_LOCAL_DISTANDE = [[5, 0.87], [2, 0.35]]
# 超过这两个阈值，才能作为聚类中心点
DELTA_THRESHOLD = 4
RHO_THRESHOLD = 1.4
NEIGHBOUR_DISTANCE = 10


def read_voxels(path1, path2, path3):
    ground_list = []
    with open(path1, mode='r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if not len(line) or line.startswith('#'):
                continue
            items = line.split(' ')
            ground_list.append([int(items[0]), int(items[1]), int(items[2])])
    points_count_list, intensity_list, distance_list = [], [], []
    none_ground_list = []
    with open(path2, 'r') as t_file:
        f_in = t_file.readlines()
        for one_line in f_in:
            one_line = one_line.strip()
            if not len(one_line) or one_line.startswith('#'):       # 判断是否是空行或注释行
                continue
            r_items = one_line.split(' ')
            none_ground_list.append([int(r_items[0]), int(r_items[1]), int(r_items[2])])
            points_count_list.append(int(r_items[3]))
            intensity_list.append(float(r_items[4]))
            distance_list.append(float(r_items[5]))
    index_list = []
    with open(path3, mode='r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if not len(line) or line.startswith('#'):
                continue
            index_list.append(int(line))

    return np.array(ground_list), np.array(none_ground_list), points_count_list, intensity_list, distance_list, index_list


def voxelization(infile_path, voxel_size):
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
        intensity_in_one_voxel_array.append(round(intensity_in_one_voxel_count / float(points_in_one_voxel_count), 2))
        points_in_one_voxel_array.append(points_in_one_voxel_count)
        # save the code to an array which later will be stored in the csv file
        voxel_count += 1
    dataset = np.vstack([np.array(coordinate_list)[:, 0], np.array(coordinate_list)[:, 1], np.array(coordinate_list)[:, 2]]).transpose()
    return dataset, points_in_one_voxel_array, intensity_in_one_voxel_array


def ground_detection(dataset):
    """
    地面点的提取

    通过从最低位置点开始，向上增长，如果没有向上点或者向上点不够，则作为潜在地面点，再通过其他的限制条件比如法向量，维度等条件再筛选
    """
    widgets = ['ground_detection: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='>'), ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=100).start()
    seeds_list = []
    voxel_length = len(dataset)
    count = 0
    # 计算出所有种子
    while count < voxel_length - 1:  # 每次循环到下一个水平位置点
        continue_flag = True
        temp_seeds = [count]  # 存储的某一位置点连续向上的体素集合
        count += 1
        if count < voxel_length:
            # 与前一个体素x,y 不相同，找的是最低高度的体素
            while dataset[:, 0][count] == dataset[:, 0][count - 1] and dataset[:, 1][count] == dataset[:, 1][count - 1]:
                if dataset[:, 2][count] - dataset[:, 2][count - 1] <= int(CONNECTIVITY_THRESHOLD / VOXEL_SIZE) and continue_flag:
                    temp_seeds.append(count)
                    if count >= voxel_length - 1:
                        break
                else:
                    continue_flag = False
                count += 1
                if count >= voxel_length:
                    break
            if len(temp_seeds) <= GROUND_HEIGHT / VOXEL_SIZE:
                for seed in temp_seeds:
                    seeds_list.append(seed)
    if dataset[:, 0][voxel_length - 1] != dataset[:, 0][voxel_length - 2] or dataset[:, 1][voxel_length - 1] != \
            dataset[:, 1][voxel_length - 2]:
        seeds_list.append(voxel_length - 1)
    pbar.update(20)
    new_dataset = np.vstack([dataset[:, 0], dataset[:, 1]]).transpose()  # 投影三维数据到二维平面，构建二维的数据集
    new_tree = scipy.spatial.cKDTree(new_dataset)
    final_ground_seeds = []
    # ##判断是否局部处于底部区域1.默认最大坡度15°。2米范围内的高差是2 * sin(15°) = 0.52。如果该点与周围点中最低点距离大于这个值不被认为是地面点
    # ##2. 如果该点与平均值相差太远也不属于地面点
    count = 0
    for new_seed in seeds_list:
        count += 1
        pbar.update(20 + 80.0 * count / len(seeds_list))
        ground_flag = True
        for item in GROUND_LOCAL_DISTANDE:
            neighbors = new_tree.query_ball_point(new_dataset[new_seed], item[0] / VOXEL_SIZE)
            minz = min(dataset[:, 2][neighbors])
            if dataset[:, 2][new_seed] - minz > (item[1] / VOXEL_SIZE < 1 and 1 or int(item[1] / VOXEL_SIZE + 0.5)):  # 小于1当作1
                ground_flag = False
                break
        if ground_flag:
            final_ground_seeds.append(new_seed)
    pbar.update(100)
    # ##判断位置点与地面点的距离，以排除一些非位置点
    pbar.finish()
    return final_ground_seeds


def get_distance(pt1, pt2):
    tmp = (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2
    return tmp ** 0.5


def get_compund_distance(pt1, pt2, i1, i2):
    tmp = pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2) + pow(pt1[2] - pt2[2], 2)
    u_distance = pow(tmp, 0.5)
    i_distance = math.fabs(i1 - i2)
    return u_distance + i_distance


def get_distance_to_ground(ground, none_ground):
    """
    密度计算函数

    此处的密度计算公式是 rho(p) = h - n / h + c/100, 其中h为当前点的z方向点累计体素数，n为当前点z方向的计数，c为这一位置点数
    比如(3,2,5)这个点，(3,2)位置有10个体素，80个点，则rho(p) = 10 - 5/10 + 0.8 = 10.3
    """
    ground_tree = scipy.spatial.cKDTree(ground)  # ground
    voxel_length = len(none_ground)
    count = 0
    distance_to_ground_list = [0] * voxel_length
    # 计算出所有种子
    while count < voxel_length:  # 每次循环到下一个水平位置点
        temp_seeds = [count]  # 存储的是一个水平位置点的所有体素
        # 垂直方向点计数
        count += 1
        # 垂直高度计数
        h_count = 1
        # 计算某一水平位置的体素集合
        if count < voxel_length:
            while none_ground[:, 0][count] == none_ground[:, 0][count - 1] and none_ground[:, 1][count] == none_ground[:, 1][count - 1]:
                h_count += 1
                temp_seeds.append(count)
                count += 1
                if count >= voxel_length:
                    break
        distance_to_ground, neighbor_index = ground_tree.query(none_ground[temp_seeds[0]])
        for seed in temp_seeds:  # 同一垂直线上的到地面的距离一样
            distance_to_ground_list[seed] = math.fabs(ground[:, 2][neighbor_index] - none_ground[:, 2][temp_seeds[0]])
    return distance_to_ground_list


def set_rho_delta(dataset, point_count_list, intensity_list):
    """
    密度计算函数

    此处的密度计算公式是 rho(p) = h - n / h + c/100, 其中h为当前点的z方向点累计体素数，n为当前点z方向的计数，c为这一位置点数
    比如(3,2,5)这个点，(3,2)位置有10个体素，80个点，则rho(p) = 10 - 5/10 + 0.8 = 10.3
    """
    widgets = ['clustering: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='>'), ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=100).start()
    none_ground_tree = scipy.spatial.cKDTree(dataset)  # none_ground
    voxel_length = len(dataset)
    # 记录每个体素的密度
    rho_list = [-1] * voxel_length
    # 记录每个体素到更高密度体素的最小距离
    delta_list = [-1] * voxel_length
    # 记录一定范围内每个体素密度高于当前体素的最近体素编号
    nearest_neighbor_list = [-1] * voxel_length
    count = 0
    intensity_range = float(max(intensity_list) - min(intensity_list))
    # 计算出所有种子
    while count < voxel_length:  # 每次循环到下一个水平位置点
        pbar.update(20 * count / voxel_length)
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
            # todo  密度函数,修改rho计算函数，不再和z直接相关,而是和连续向上的高度相关
            # rho_list[seed] = ((h_count - float(temp_count) / h_count) + point_count / 10000.0) / (distance_to_ground_list[seed] + 1)
            rho_list[seed] = ((h_count - float(temp_count) / h_count) + point_count / 10000.0) / (dataset[:, 2][seed] + 1)
            delta = 0
            if temp_count == 0:
                temp_count += 1
                continue
            # 与密度较高体素的距离
            delta = dataset[:, 2][seed] - dataset[:, 2][seed - 1]
            # 这里改为delta>=1就可以取消上下分类必须一致的限制，但是这一操作会大量增加计算量，所以不考虑这种方法，而改小体素大小
            if delta > 1:
                temp_count += 1
                continue
            nearest_neighbor_list[seed] = seed - 1
            delta_list[seed] = delta
            temp_count += 1
    pbar.update(20)
    for ii in range(voxel_length):
        pbar.update(20 + 70.0 * ii / voxel_length)
        if delta_list[ii] != -1:
            continue
        neighbors = none_ground_tree.query_ball_point(dataset[ii], NEIGHBOUR_DISTANCE)
        distance_list = []
        neighbor_list = []
        for neighbor in neighbors:
            # 计算比当前点密度大到该点的距离
            if rho_list[neighbor] > rho_list[ii]:
                distance = get_distance(dataset[ii], dataset[neighbor])
                #  把intensity加入距离计算之中
                # compund_distance = distance + 4 *  math.fabs((intensity_list[ii] - intensity_list[neighbor])) / intensity_range
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
    pbar.update(90)
    for ii in range(len(rho_sorted)):
        id_p = rho_sorted[ii][1]
        if delta_list[id_p] > DELTA_THRESHOLD and rho_list[id_p] > RHO_THRESHOLD:
            class_list[id_p] = class_num
            class_num += 1
        else:
            if class_list[nearest_neighbor_list[id_p]] != -1:
                class_list[id_p] = class_list[nearest_neighbor_list[id_p]]
            else:
                class_list[id_p] = 0  # 异常点的类别设为0，如rho较小但是delta比较大的点
    pbar.update(100)
    pbar.finish()
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
    infilepath = 'data/lyx2.las'
    outtxt = infilepath[:-4] + '_' + str(VOXEL_SIZE) + '.txt'
    outtxt1 = infilepath[:-4] + '_' + str(VOXEL_SIZE) + '_ground' + '.txt'
    outtxt2 = infilepath[:-4] + '_' + str(VOXEL_SIZE) + '_index' + '.txt'
    print "\n1. voxelizing... "
    if not os.path.exists(outtxt):
        original_dataset, points_count_array, intensity_array = voxelization(infilepath, VOXEL_SIZE)
        print "\n2. ground detecting..."
        ground_index = ground_detection(original_dataset)
        ground_dataset = original_dataset[ground_index]
        none_ground_index = range(len(original_dataset))
        for item in sorted(ground_index, reverse=True):
            none_ground_index.pop(item)
        none_ground_dataset = original_dataset[none_ground_index]
        # distance_to_ground_list = get_distance_to_ground(ground_dataset, none_ground_dataset)
        with open(outtxt1, mode='w') as fout:
            for i in ground_dataset:
                line = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n'
                fout.write(line)
        with open(outtxt2, mode='w') as fout:
            for i in none_ground_index:
                line = str(i) + '\n'
                fout.write(line)
    else:
        ground_dataset, none_ground_dataset, points_count_array, intensity_array, distance_to_ground_list, none_ground_index = read_voxels(outtxt1, outtxt, outtxt2)
    print "\n3. clustering..."
    # neighbors, rhos, deltas, classes = set_rho_delta(none_ground_dataset, points_count_array, intensity_array, distance_to_ground_list)
    neighbors, rhos, deltas, classes = set_rho_delta(none_ground_dataset, points_count_array, intensity_array)
    with open(outtxt, 'w') as fout:
        for x, y, z, p, i, d, neighbor, rhoitem, deltaitem, classitem in zip(none_ground_dataset[:, 0], none_ground_dataset[:, 1], none_ground_dataset[:, 2], points_count_array, intensity_array, distance_to_ground_list, neighbors, rhos, deltas, classes):
            line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(p) + ' ' + str(i)
            line = line + ' ' + str("%.2f" % d) + ' ' + str(neighbor) + ' ' + str("%.2f" % rhoitem) + ' ' + str("%.2f" % deltaitem) + ' ' + str(classitem) + '\n'
            fout.write(line)
    lasfile = laspy.file.File(infilepath, mode="rw")
    point_length = len(lasfile.x)
    point_count = 0
    print "\n4.labeling points..."
    widgets = ['labeling: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='>'), ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=1).start()
    percent_start = 1.0 / len(lasfile.gps_time)
    percent = 0
    # 构造classlist反对应到点云
    lasfile.pt_src_id[:] = 0
    none_ground_index_array = np.array(none_ground_index)
    for time in lasfile.gps_time:
        index = np.where(none_ground_index_array == time)[0]
        if len(index) == 0:
            lasfile.pt_src_id[point_count] = 0
        else:
            lasfile.pt_src_id[point_count] = classes[index[0]]
        point_count += 1
        percent += percent_start
        pbar.update(percent)
    pbar.update(1)
    pbar.finish()
    lasfile.close()

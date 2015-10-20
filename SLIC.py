# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import geometry
import scipy.spatial
import numpy as np
import scipy.linalg as la
import laspy
import yylog


RADIUS = 0.6
GRID_STEP = 0.7
ITERATION_NUMBER = 10


def distance(point1, point2):
    pass


def setdimensionality(infile):
    geometry.write_optimal_local_information(infile)


def preprocess(infile):
    """
    preprocess of point cloud

    points that are isolated are removed from the original point cloud
    """
    pass


def calculate_distance(vector1, vector2):
    """
    calculate the distance between to vector corresponding to the distance between two points

    Distance is defined by users
    """
    d = np.sqrt(sum(np.power(np.array(vector1[0:3]) - np.array(vector2[0:3]), 2)))
    d /=  (2 * GRID_STEP)
    return np.sqrt(sum(np.power(np.array([vector1[3], vector1[4]]) - np.array([vector2[3], vector2[4]]), 2), d ** 2))


def slic(laspath):
    """
    SLIC supervoxel segmentation based on normal&density distance

    Args:
        points_list: points in point cloud
    """
    # ##### Initialization
    # 1.Initialize cluster centers [x, y, z, n, d]
    lasfile = laspy.file.File(laspath)
    # outfile_path = laspath.replace('.las', '_slic.las')
    # outfile = laspy.file.File(outfile_path, mode='w', header=lasfile.header)
    # outfile.define_new_dimension('cluster', 5, 'cluster number')
    # for dimension in lasfile.point_format:
    #     data = lasfile.reader.get_dimension(dimension.name)
    #     outfile.writer.set_dimension(dimension.name, data)
    # lasfile.close()
    dataset = np.vstack([lasfile.x, lasfile.y, lasfile.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    point_length = len(lasfile.x)
    minx = lasfile.header.min[0]
    miny = lasfile.header.min[1]
    minz = lasfile.header.min[2]
    # 计算Grid的个数
    x_length = int((lasfile.header.max[0] - minx) / GRID_STEP) + 1
    y_length = int((lasfile.header.max[1] - miny) / GRID_STEP) + 1
    z_length = int((lasfile.header.max[2] - minz) / GRID_STEP) + 1
    max_d = 0
    center_list = []
    indices_list = []
    normal_lsit = [0] * point_length
    density_list = [0] * point_length

    def update_cluster_centers():
        sum_list = [0] * 7
        for center_count in range(len(center_list)):
            indices = indices_list[center_count]  # 中心点周围2S 范围内的点的索引
            if len(indices) == 0:
                continue
            for indice in indices:
                element_count = 0  # 记录属于这个中心点对应聚类的点的个数
                if label_list[indice] == center_count:
                    element_count += 1
                    sum_list = [x + y for x, y in zip(sum_list, [lasfile.x[indice], lasfile.y[indice], lasfile.z[indice], normal_lsit[indice], density_list[indice]])]
            # center_list[center_count] = sum_list / element_count  # 更新当前聚类中心点
            if element_count == 0:
                continue
            for i in range(len(center_list[center_count])):
                center_list[center_count][i] = sum_list[i] / element_count
            center_count += 1

    for i in range(x_length):
        for j in range(y_length):
            for k in range(z_length):
                # 计算GRID center的x y z坐标
                x = i * GRID_STEP + minx
                y = j * GRID_STEP + miny
                z = k * GRID_STEP + minz
                indices = tree.query_ball_point([x, y, z], RADIUS)
                if len(indices) < 3:
                    continue
                if len(indices) > max_d:
                    max_d = len(indices)
                data = np.vstack([lasfile.x[indices], lasfile.y[indices], lasfile.z[indices]]).transpose()
                cov = np.cov(data)
                eign_values, eign_vectors = la.eig(cov)
                index = eign_values.argsort()[::-1]
                eign_vectors = eign_vectors[:, index]
                n = abs(eign_vectors[2][2])
                d = len(indices)
                center_list.append([x, y, z, n, d])
    for elem in center_list:
        elem[4] /= float(max_d)
    # 2.Initialize labels and distances
    label_list = [-1] * point_length
    distance_list = [float('inf')] * point_length
    # 3.repeating of labeling points around a center
    point_count = 0
    while point_count < point_length:
        indices = tree.query_ball_point([lasfile.x[point_count], lasfile.y[point_count], lasfile.z[point_count]], RADIUS)
        if len(indices) <= 3:
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        evals, evects = la.eigh(cov)
        evals = np.abs(evals)
        index = evals.argsort()[::-1]
        evects = evects[:, index]
        normals = evects[2]
        normal_lsit[point_count] = abs(normals[2])
        density_list[point_count] = len(indices) / float(max_d)
        point_count += 1
    while True:
        iterated_time = 0  # 循环次数
        center_count = 0
        indices_list = []
        for center in center_list:
            indices = tree.query_ball_point([center[0], center[1], center[2]], 2 * RADIUS)
            indices_list.append(indices)
            for indice in indices:
                current_vector = [lasfile.x[indice], lasfile.y[indice], lasfile.z[indice],normal_lsit[indice], density_list[indice]]
                dis = calculate_distance(center, current_vector)
                if dis < distance_list[indice]:
                    distance_list[indice] = dis
                    label_list[indice] = center_count
            center_count += 1
        # 4. updating cluster centers
        update_cluster_centers()
        iterated_time += 1
        if iterated_time > ITERATION_NUMBER:
            break
    label_count = 0
    for label in label_list:
        lasfile.cluster[label_count] = label
    lasfile.close()

if __name__ == '__main__':
    import os
    loop = True
    while loop:
        path = raw_input('\n Now please input the original file name: \n')
        if path[-4:] == '.las':
            loop = False
        else:
            print("Please input a *.xyz_label_conf file!!!")
    # setdimensionality(path)
    log = yylog.LOG('')
    try:
        slic(path)
    except:
        log.error()  # 使用系统自己的错误描述
        os.system('pause')
        exit()

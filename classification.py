# -*- coding: UTF-8 -*-
__author__ = 'Administrator'

import laspy
import numpy as np
import scipy.spatial
import struct
import liblas
import math
import time
import os
import csv
import __future__
# from matplotlib.mlab import PCA
# import matplotlib.pyplot as plt


def entropy_function(dimensions):
    import math

    if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] <= 0:  #
        return 3.40e+38
    else:
        return -dimensions[0] * math.log(dimensions[0]) - dimensions[1] * math.log(dimensions[1]) - dimensions[2] * math.log(dimensions[2])


def get_dimensions(dataset, tree, x, y, z, radius):
    """
    return the dimensionality values of a point within a radius

    the returned values are stored in a list in sequence,
    they 1d value, 2d value, 3d value respectivesly

    Args:
        dataset: the input x,y,z data of all points
        tree: kdtree build upon dataset
        x,y,z: the coordinate of the point to get dimensionality
        radius: the radius to define the dimensionality

    Returns:
        a list which stores the 1d value, 2d value, 3d value in order

    Raises:

    """
    import math

    evals = get_descent_eignvalues(dataset, tree, x, y, z, radius)
    if evals is None:
        return
    mu1 = math.sqrt(evals[0])
    mu2 = math.sqrt(evals[1])
    mu3 = math.sqrt(evals[2])
    # 出现重复点也可能导致mu1为0，让它的熵值最大，然后就可以继续选点；
    if mu1 == 0:
        return
    a1d, a2d, a3d = 1.0 - mu2 / mu1, mu2 / mu1 - mu3 / mu1, mu3 / mu1
    return [a1d, a2d, a3d]


def get_optimal_radius(dataset, kdtree, x, y, z, rmin, rmax, delta):
    """
    通过计算最小熵值，来求最优临近半径

    Args:
        dataset: 点所在的数据集
        kdtree： dataset对应的kdtree
        x,y,z: 待求点
        rmin： 最小半径
        rmax： 允许的最大半径
        delta： 步进大小
    """
    rtemp = rmin
    dimensions = get_dimensions(dataset, kdtree, x, y, z, rtemp)
    if dimensions is None:
        ef = 3.40282e+038
    else:
        ef = entropy_function(dimensions)
    efmin = ef
    rotpimal = rmin
    rtemp += delta
    count = 1
    while rtemp < rmax:
        dimensions = get_dimensions(dataset, kdtree, x, y, z, rtemp)
        # 按e**（0.12*count**2)递增
        # rtemp += 2.71828 ** (0.12 * count * count) * deltar
        rtemp += 0.08 * count
        count += 1
        if dimensions is None:
            continue
        ef = entropy_function(dimensions)
        if ef < efmin:
            efmin = ef
            rotpimal = rtemp
    return rotpimal


def get_descent_eignvalues(dataset, tree, x, y, z, radius):
    """
    计算数据的对应的维度值1d,2d,3d,并按由大到小顺序返回
    """
    from scipy import linalg as la
    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:  # 邻居点少于三个的情况，计算不了协方差矩阵和特征值。让它的熵值最大，然后就可以继续选点；
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals = la.eigh(cov, eigvals_only=True)  # #如果用通常用的la.eig 函数的话特征值会出现复数
    evals = np.abs(evals)  # 因为超出精度范围，所以值有可能出现负数，这里折中取个绝对值，因为反正都很小，可以忽略不计
    index = evals.argsort()[::-1]
    # evects=evects[:,index]
    evals = evals[index]
    return evals


def get_normal(dataset, tree, x, y, z, radius):
    from scipy import linalg as la

    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals, evects = la.eigh(cov)
    evals = np.abs(evals)
    index = evals.argsort()[::-1]
    evects = evects[:, index]
    return evects[2]


def get_eigenvectors(dataset, tree, x, y, z, radius):
    from scipy import linalg as la

    indices = tree.query_ball_point([x, y, z], radius)
    if len(indices) <= 3:
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals, evects = la.eigh(cov)
    evals = np.abs(evals)
    index = evals.argsort()[::-1]
    evects = evects[:, index]
    return evects


def feature_extraction(dataset, kdtree):
    """
    提取dataset中每个点的特征

    提取的信息存储在一个二维数组中返回
    """
    from progressbar import *
    widgets = ['Extracting features: ', Percentage(), ' ', Bar(marker=RotatingMarker('>-=')), ' ', Timer(), ' ']
    p = ProgressBar(widgets=widgets).start()
    length = len(dataset)
    count = 0
    features = []
    while count < length:
        x, y, z = dataset[count][0], dataset[count][1], dataset[count][2]
        optimal_radius = get_optimal_radius(dataset, kdtree, x, y, z, 0.08, 0.8, 0.08)
        evals = get_descent_eignvalues(dataset, kdtree, x, y, z, optimal_radius)
        # 出现极端情况则让所有特征为0
        if evals is None:
            feature_3d = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            e1, e2, e3 = evals[0], evals[1], evals[2]
            L = (e1 - e2) / e1
            P = (e2 - e3) / e1
            S = e3 / e1
            O = (e1 * e2 * e3) ** (1.0 / 3)
            A = (e1 - e3) / e1
            E = -e1 * math.log(e1) - e2 * math.log(e2) - e3 * math.log(e3 + 1e-100)
            Sum = e1 + e2 + e3
            C = e3 / (e1 + e2 + e3)
            feature_3d = [L, P, S, O, A, E, Sum, C]
        features.append(feature_3d)
        count += 1
        p.update(int((count / (length - 1.0)) * 100))
    p.finish()
    return features


def write_features(outfile_name, dataset, kdtree, class_list):
    """
    提取特征，将特征写入csv文件

    Args:
        infilepath: the path of the file to be written
    """
    import os
    start = time.clock()
    # the name of output file
    extracted_features = feature_extraction(dataset, kdtree)
    with open(outfile_name, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        writer.writerow([ 'L', 'P', 'S', 'O', 'A', 'E', 'Sum', 'C', 'class'])
        for feature, class_id in zip(extracted_features, class_list):
            writer.writerow([feature[0], feature[1], feature[2], feature[3], feature[4], feature[5], feature[6],
                             feature[7], class_id])

loop = True
while loop:
    path = raw_input('\n Now please input the original file name: \n')
    if path[-15:] == '.xyz_label_conf':
        loop = False
    else:
        print("Please input a *.xyz_label_conf file!!!")
out_csv = path[:-15] + '.csv'
file = open(path)
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
dataset = np.vstack([x_list, y_list, z_list]).transpose()
tree = scipy.spatial.cKDTree(dataset)
write_features(out_csv, dataset, tree, class_list)
os.system('pause')

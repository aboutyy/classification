# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import math
import laspy
import numpy as np
import geometry
import scipy.spatial
import os
import progressbar


def region_growing(voxelset, normal_list, planarity_list, radius, angle_threshold):
    # codes below were region growing algorithm implemented based pseudocode in
    # http://pointclouds.org/documentation/tutorials/region_growing_segmentation.php#region-growing-segmentation
    # Point cloud: voxelset
    # Point Normals: normal_list
    # Angle threshold: angle_threshold
    # Awailable point list: a_list
    widgets = ['region growing: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='>'), ' ', progressbar.Timer(), ' ']
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=100).start()
    tree = scipy.spatial.cKDTree(voxelset)
    length = len(voxelset)
    a_list = range(length)
    seed_length = len(a_list)
    # region list
    regions = []
    # 种子点，只有法向量基本平行于地面的可被当作种子点
    available_seed_list = list(np.where(np.logical_and(np.array(planarity_list) > 0.3, np.array(normal_list) < 0.3))[0])
    length = len(available_seed_list)
    while len(available_seed_list) > 0:
        current_region = []
        current_seeds = []
        # voxel with lowest z value
        lowest_voxel_indice = available_seed_list[0]
        current_seeds.append(lowest_voxel_indice)
        current_region.append(lowest_voxel_indice)
        available_seed_list.remove(lowest_voxel_indice)
        a_list.remove(lowest_voxel_indice)
        count = 0
        while count < len(current_seeds):
            current_seed = current_seeds[count]
            count += 1
            current_seed_neighbors = tree.query_ball_point([voxelset[:, 0][current_seed], voxelset[:, 1][current_seed],
                                                            voxelset[:, 2][current_seed]], radius)
            for neighbor in current_seed_neighbors:
                if a_list.count(neighbor) != 0:
                    if current_region.count(neighbor) == 0 and \
                                    math.fabs(math.acos(math.fabs(normal_list[neighbor])) -
                                                      math.acos(math.fabs(normal_list[current_seed]))) < angle_threshold:
                        current_region.append(neighbor)
                        a_list.remove(neighbor)
                        if current_seeds.count(neighbor) == 0 and available_seed_list.count(neighbor) != 0:
                            current_seeds.append(neighbor)
                            available_seed_list.remove(neighbor)
                            pbar.update(100 * (1 - len(available_seed_list) / float(length)))
        regions.append(np.array(current_region))
    pbar.finish()
    return regions


def write_geometry_features(in_path, radius):
    """
    写入几何特征到las文件中，包括Nx, Ny, Nz 和 planarity

    ok
    """

    in_file = laspy.file.File(in_path, mode='rw')
    dataset = np.vstack([in_file.x, in_file.y, in_file.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    count = 0
    for x, y, z in zip(in_file.x, in_file.y, in_file.z):
        eignvalues = geometry.get_descent_eignvalues(dataset, tree, x, y, z, radius)
        if eignvalues is None:  # 如果是孤立点，则不当作是平面点
            in_file.planarity[count] = 0
        else:
            in_file.planarity[count] = (eignvalues[1] - eignvalues[2]) / eignvalues[0]
        normals = geometry.get_normals(dataset, tree, x, y, z, radius)
        if normals is None:  # 如果孤立点则当作是垂直地面的点
            in_file.f1[count] = 0
            in_file.f2[count] = 0
            in_file.f3[count] = 1
        else:
            in_file.f1[count] = math.fabs(normals[0])
            in_file.f2[count] = math.fabs(normals[1])
            in_file.f3[count] = math.fabs(normals[2])
        count += 1
    in_file.close()


def preprocessing(in_path, out_path, radius):
    out_path1 = in_path[:-4] + '-t' + '.las'
    geometry.clean_isolated_points(in_path, out_path1, 0.5, 4)
    geometry.add_dimention(out_path1, out_path, ['planarity', 'f1', 'f2', 'f3'], [9, 9, 9, 9], ['lambda1-lambda3/lambda1', 'extra field1', 'extra field2', 'extra field3'])
    os.remove(out_path1)
    write_geometry_features(out_path, radius)


def segment_merging(regions, distance):
    pass

if __name__ == '__main__':
    in_path = '5.las'
    out_path = in_path[:-4] + '-p' + '.las'
    radius = 0.5
    if not os.path.exists(out_path):
        preprocessing(in_path, out_path, radius)
    file = laspy.file.File(out_path, mode='rw')
    dataset = np.vstack([file.x, file.y, file.z]).transpose()
    regions = region_growing(dataset, file.f3, file.planarity, 0.5, 0.15)
    file.raw_classification[:] = 0
    count = 1
    for region in regions:
        if len(region) > 50:
            file.raw_classification[region] = count
            count += 1

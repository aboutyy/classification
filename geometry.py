# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import laspy
import numpy as np
import scipy.spatial
import time
import csv

OPTIMAL_DIMENSION_NAME = 'optimal_dimensionalities'
OPTIMAL_NX_NAME = 'optimal_nx'
OPTIMAL_NY_NAME = 'optimal_ny'
OPTIMAL_NZ_name = 'optimal_nz'
OPTIMAL_PX_NAME = 'optimal_px'
OPTIMAL_PY_NAME = 'optimal_py'
OPTIMAL_PZ_NAME = 'optimal_pz'
OPTIMAL_RADIUS_NAME = 'optimal_radius'


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
    """
    计算Nz的值

    计算[x,y,z]点的x,y平面的垂直方向法向量值

    """

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
    return evects[2][2]


def get_normal(dataset, tree, point, radius):
    """
    计算Nz的值

    计算[x,y,z]点的x,y平面的垂直方向法向量值

    """

    from scipy import linalg as la
    indices = tree.query_ball_point(point, radius)
    if len(indices) <= 3:
        return
    idx = tuple(indices)
    data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
    cov = np.cov(data)
    evals, evects = la.eigh(cov)
    evals = np.abs(evals)
    index = evals.argsort()[::-1]
    evects = evects[:, index]
    return evects[2][2]


def get_normals(dataset, tree, x, y, z, radius):
    """
    计算Nz的值

    计算[x,y,z]点的x,y平面的垂直方向法向量值

    """

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


def entropy_function(dimensions):
    import math

    if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] <= 0:  #
        return 3.40e+38
    else:
        return -dimensions[0] * math.log(dimensions[0]) - dimensions[1] * math.log(dimensions[1]) - dimensions[
                                                                                                        2] * math.log(
            dimensions[2])


def get_optimal_radius(dataset, kdtree, x, y, z, rmin, rmax, deltar):
    """
    通过计算最小熵值，来求最优临近半径
    """
    rtemp = rmin
    rotpimal = rmin
    count = 1
    efmin = 3.40282e+038
    while rtemp < rmax:
        dimensions = get_dimensions(dataset, kdtree, x, y, z, rtemp)
        # 按e**（0.12*count**2)递增
        # rtemp += 2.71828 ** (0.12 * count * count) * deltar
        if dimensions is None:
            # rtemp += 0.08 * count
            # count += 1
            rtemp += 0.08
            continue
        ef = entropy_function(dimensions)
        if ef < efmin:
            efmin = ef
            rotpimal = rtemp
        # rtemp += 0.08 * count
        # count += 1
        rtemp += 0.08
    return rotpimal


def write_normals(infilepath, outfilepath, radius):
    """
    write the normals(nx, ny, nz) value of points in a file to a new field

    if the field does not exist, add a new field named normal
    """
    from scipy import linalg as la

    infile = laspy.file.File(infilepath, mode='rw')
    outfile = laspy.file.File(outfilepath, mode='w', header=infile.header)
    dataset = np.vstack((infile.x, infile.y, infile.z)).transpose()
    kd_tree = scipy.spatial.cKDTree(dataset)
    count = 0
    for x, y, z in zip(infile.x, infile.y, infile.z):
        indices = kd_tree.query_ball_point([x, y, z], radius)
        # 邻居点少于三个的情况，计算不了协方差矩阵和特征值。让它的熵值最大，然后就可以继续选点；
        if len(indices) <= 3:
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        eign_values, eign_vectors = la.eig(cov)
        index = eign_values.argsort()[::-1]
        eign_vectors = eign_vectors[:, index]
        infile.gps_time[count] = eign_vectors[2][2]
        count += 1
        print count
    infile.close()
    print 'Write %d Normal values successfully!' % count


def filter_scatter_points(infile_path, outfile_path, radius, point_count):
    """
    去除点云文件中的离散点

    radius范围内如果少于point_count个点，这个点就属于离散点
    """
    infile = laspy.file.File(infile_path, mode='r')
    out_indices = []
    outfile = laspy.file.File(outfile_path, mode='w', header=infile.header)
    dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    count = 0
    for x, y, z in zip(infile.x, infile.y, infile.z):
        indices = tree.query_ball_point([x, y, z], radius)
        if len(indices) >= point_count:
            out_indices.append(count)
        count += 1
        print count
    outfile.points = infile.points[out_indices]
    outfile.close()
    infile.close()
    print 'Filter done!'


def write_optimal_local_information(infilepath):
    """
    write the dimensionality and normals and principle directions

    write the dimensionalities, normals, principle directions of all points in a file
    to new fields when the radius is optimal
    Args:
        infilepath: the path of the file to be written
    """
    import os

    start = time.clock()
    # the name of output file
    outfile_path = infilepath.replace('.las', '_optimal.las')
    infile = laspy.file.File(infilepath, mode='r')
    outfile = laspy.file.File(outfile_path, mode='w', header=infile.header)
    outfile.define_new_dimension(OPTIMAL_RADIUS_NAME, 9, 'optimal radius')
    outfile.define_new_dimension(OPTIMAL_DIMENSION_NAME, 1, 'dimensionality with optimal radius')

    outfile.define_new_dimension(OPTIMAL_NX_NAME, 9, 'normals nx with optimal radius')
    outfile.define_new_dimension(OPTIMAL_NY_NAME, 9, 'normals ny with optimal radius')
    outfile.define_new_dimension(OPTIMAL_NZ_name, 9, 'normals nz with optimal radius')

    outfile.define_new_dimension(OPTIMAL_PX_NAME, 9, 'principle directions px with optimal radius')
    outfile.define_new_dimension(OPTIMAL_PY_NAME, 9, 'principle directions py with optimal radius')
    outfile.define_new_dimension(OPTIMAL_PZ_NAME, 9, 'principle directions pz with optimal radius')
    for dimension in infile.point_format:
        data = infile.reader.get_dimension(dimension.name)
        outfile.writer.set_dimension(dimension.name, data)
    dataset = np.vstack([outfile.x, outfile.y, outfile.z]).transpose()
    kdtree = scipy.spatial.cKDTree(dataset)
    print len(outfile.points)
    length = len(outfile.points)
    count = 0
    try:
        while count < length:
            x, y, z = outfile.x[count], outfile.y[count], outfile.z[count]
            optimal_radius = get_optimal_radius(dataset, kdtree, x, y, z, 0.1, 0.6, 0.08)
            outfile.optimal_radius[count] = optimal_radius
            eigenvectors = get_eigenvectors(dataset, kdtree, x, y, z, optimal_radius)
            if eigenvectors is None:
                count += 1
                continue
            outfile.optimal_nx[count] = eigenvectors[2][0]
            outfile.optimal_ny[count] = eigenvectors[2][1]
            outfile.optimal_nz[count] = eigenvectors[2][2]
            outfile.optimal_px[count] = eigenvectors[0][0]
            outfile.optimal_py[count] = eigenvectors[0][1]
            outfile.optimal_pz[count] = eigenvectors[0][2]
            dimensions = get_dimensions(dataset, kdtree, x, y, z, optimal_radius)
            # if the point has no dimension values it means it doesn't have enough neighbouring points
            if dimensions is None:
                outfile.optimal_dimensionalities[count] = 3
            else:
                dimension = max(dimensions[0], dimensions[1], dimensions[2])
                if dimensions[0] == dimension:
                    outfile.optimal_dimensionalities[count] = 1
                elif dimensions[1] == dimension:
                    outfile.optimal_dimensionalities[count] = 2
                elif dimensions[2] == dimension:
                    outfile.optimal_dimensionalities[count] = 3
            count += 1
            if count % 100 == 0:
                print count
    except:
        print time.clock() - start
        print "Wrong"
        time.sleep(1000)
    else:
        infile.close()
        outfile.close()
        print time.clock() - start
        print 'Done!'
        os.system("pause")


def add_dimention(infile_path, outfile_path, names, types, descriptions):
    """
    add new dimensions to the las file

    Args:
        names: names array of the dimensions
        types: types array of the dimensions
                0	Raw Extra Bytes	Value of “options”
                1	unsigned char	1 byte
                2	Char	1 byte
                3	unsigned short	2 bytes
                4	Short	2 bytes
                5	unsigned long	4 bytes
                6	Long	4 bytes
                7	unsigned long long	8 bytes
                8	long long	8 bytes
                9	Float	4 bytes
                10	Double	8 bytes
                11	unsigned char[2]	2 byte
                12	char[2]	2 byte
                13	unsigned short[2]	4 bytes
                14	short[2]	4 bytes
                15	unsigned long[2]	8 bytes
                16	long[2]	8 bytes
                17	unsigned long long[2]	16 bytes
                18	long long[2]	16 bytes
                19	float[2]	8 bytes
                20	double[2]	16 bytes
                21	unsigned char[3]	3 byte
                22	char[3]	3 byte
                23	unsigned short[3]	6 bytes
                24	short[3]	6 bytes
                25	unsigned long[3]	12 bytes
                26	long[3]	12 bytes
                27	unsigned long long[3]	24 bytes
                28	long long[3]	24 bytes
                29	float[3]	12 bytes
                30	double[3]	24 bytes
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


def write_normal(in_file, out_file, fixed_radius):
    import scipy
    from scipy import linalg as la

    with open(in_file, 'rb') as incsv:
        reader = csv.reader(incsv)
        lines = [[row[0], row[1], row[2], row[3], row[4]] for row in reader]
    original_code_array = np.array(lines)[:, 0]
    original_x_int_array = np.vectorize(int)(map(lambda x: x[:4], original_code_array))
    original_y_int_array = np.vectorize(int)(map(lambda x: x[4:8], original_code_array))
    original_z_int_array = np.vectorize(int)(map(lambda x: x[8:12], original_code_array))

    point_counts_array = np.array(lines)[:, 1]
    intensity_array = np.array(lines)[:, 2]
    olocation_array = np.array(lines)[:, 3]
    mlocation_array = np.array(lines)[:, 4]

    normal_list = []
    dataset = np.vstack([original_x_int_array, original_y_int_array, original_z_int_array]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    for x, y, z in zip(original_x_int_array, original_y_int_array, original_z_int_array):
        indices = tree.query_ball_point([x, y, z], fixed_radius)
        if len(indices) <= 3:
            continue
        idx = tuple(indices)
        data = np.vstack([dataset[idx, 0], dataset[idx, 1], dataset[idx, 2]])
        cov = np.cov(data)
        evals, evects = la.eigh(cov)
        evals = np.abs(evals)
        index = evals.argsort()[::-1]
        evects = evects[:, index]
        normal_list.append(evects[2])
    length = len(original_code_array)
    count = 0
    with open(out_file, 'wb') as out_csv:
        writer = csv.writer(out_csv)
        while count < length:
            writer.writerow([original_code_array[count], point_counts_array[count], intensity_array[count],
                             olocation_array[count], mlocation_array[count], normal_list[count]])
            count += 1


def clean_isolated_points(in_file, out_file, radius, num_neighbors):
    file1 = laspy.file.File(in_file, mode='r')
    file2 = laspy.file.File(out_file, header=file1.header, mode='w')
    dataset = np.vstack([file1.x, file1.y, file1.z]).transpose()
    tree = scipy.spatial.cKDTree(dataset)
    count = 0
    point_list = []
    for x, y, z in zip(file1.x, file1.y, file1.z):
        neighbors = tree.query_ball_point([x,y,z], radius)
        if len(neighbors) > num_neighbors:
            point_list.append(count)
        count += 1
    file2.points = file1.points[point_list]
    file1.close()
    file2.close()


"""
过滤地面点

infilepath = r"france.las"
infile = laspy.file.File(infilepath,mode='r')
outfile= laspy.file.File(r'francefore.las', mode='w',header=infile.header)
outfile1= laspy.file.File(r'franceback.las', mode='w',header=infile.header)
#找出二维平面点
planeidx = list(np.where(np.logical_and((infile.user_data==2),(np.abs(infile.gps_time)>0.866))))
backidx =  list(np.where(np.logical_or((infile.user_data!=2),(np.abs(infile.gps_time)<=0.866))))
outfile1.points = infile.points[planeidx]
outfile.points = infile.points[backidx]

outfile.close()
outfile1.close()
infile.close()

"""

"""

# 计算dimentionality

import time
import os
start=time.clock()
infilepath = r"ply2las3-clip.las"
infile=laspy.file.File(infilepath,mode='rw')
dataset = np.vstack([infile.x, infile.y, infile.z]).transpose()
kdtree=scipy.spatial.cKDTree(dataset)
print len(infile.points)
count=0
try:
    for x,y,z in zip(infile.x,infile.y,infile.z) :
        optimalradius=getoptimalradius(dataset,kdtree,x,y,z,0.1,0.58,0.08)
        print optimalradius
        infile.flag_byte[count]=int(100*optimalradius)
        a1d,a2d,a3d=getdimention(dataset,kdtree,x,y,z,optimalradius)
        dimention=max(a1d,a2d,a3d)
        if a1d==dimention:
            infile.user_data[count]=1
        elif a2d==dimention:
            infile.user_data[count]=2
        elif a3d==dimention:
            infile.user_data[count]=3
        count+=1
        print count

    # x=infile.x[3596]
    # y=infile.y[3596]
    # z=infile.z[3596]
    # optimalradius=getoptimalradius(dataset,kdtree,x,y,z,0.1,0.58,0.08)
    # a1d,a2d,a3d=getdimention(dataset,kdtree,x,y,z,optimalradius)
except:
    print time.clock()-start
    print "Wrong"
    os.system("pause")
else:
    infile.close()
    print time.clock()-start
    print 'Done!'
    os.system("pause")
"""

'''
#转换PLY为LAS

# infile= r"E:\DATA\LIDAR\France MLS Data\classified Cassette_idclass\Cassette_GT.ply"
# outfile="ply2las1.xyz"
# ply2las(infile,outfile)
laspath="ply2las3.las"
plypath=r"E:\DATA\LIDAR\France MLS Data\classified Cassette_idclass\Cassette_GT.ply"
ply2las(plypath,laspath)
# getplyinfo(plypath)
# addinfo2las(laspath,plypath)
# countclass(plypath)
'''

'''

分割

infilepath = r'E:\Thesis Experiment\Data\Test\lyx.las'
outfilepath = r'E:\Thesis Experiment\Data\Test\lyxupground.las'
#removegroud(infilepath,outfilepath,16.9)
starttime = datetime.datetime.now()
segmentation1(outfilepath,0.1,100)
endtime = datetime.datetime.now()
interval=(endtime.minute - starttime.minute)
print 'Segmentation used %d minutes' %interval
'''

'''

# 裁减las文件

infilepath = r"ply2las3 - Cloud1.las"
# shpPath = r"E:\Thesis Experiment\Data\Test\Clip1.shp"
shpPath = r"E:\MyProgram\Python\HelloWorld\shp\ply2las3 - Cloud1_aoi.shp"
inFile = laspy.file.File(infilepath, mode='r')
count = 4
polygon = readpolygonfromshp(shpPath)
inPointsIndex = getptinpolygon(inFile, polygon)
outFile = laspy.file.File(r"ply2las3-clip.las", mode='w', header=inFile.header)
outFile.points = inFile.points[inPointsIndex]
print '共裁减了%d个点' %len(outFile.points)
inFile.close()
outFile.close()
print "Done"
'''

'''

测试分类

infilepath = r"E:\class1.las"
infile=laspy.file.File(infilepath, mode='rw')
infile.red[:]=255
infile.blue[:]=0
infile.green[:]=0
a=range(10000)
b=range(10000,30001,1)
# infile.classification[:]=3
infile.raw_classification[a]=4 # 设置分类是要设置raw_classification
infile.raw_classification[b]=8
infile.classification_flags[:]=5
infile.close()
'''

'''

提取地上点

inFile = laspy.file.File(r"E:\test.las", mode='r')
upground_index=np.where(np.logical_and(inFile.z>17.1,inFile.z<50))
upground_point=inFile.points[upground_index]
outFile=laspy.file.File(r"E:\test1.las",mode='w',header=inFile.header)
inFile.close()
# plt.hist(inFile.z, 200)
# plt.show()
outFile.points=upground_point
outFile.close()
print 'Done!'
'''

'''

官方代码

#inFile = laspy.file.File("./laspytest/data/simple.las", mode = "r")
inFile = laspy.file.File(r"E:\DATA\LIDAR\Lyx\3Merged.las", mode='r')
# Grab all of the points from the file.
point_records = inFile.points
print 'Length of record is %d' % len(point_records)
# Grab just the X dimension from the file, and scale it.
def scaled_x_dimension( las_file ):
    x_dimension = las_file.X
    scale = las_file.header.scale[0]
    offset = las_file.header.offset[0]
    return ( x_dimension * scale + offset )
def scaled_z_dimension( las_file ):
    z_dimension = las_file.Z
    scale = las_file.header.scale[2]
    offset = las_file.header.offset[2]
    return ( z_dimension * scale + offset )
scaled_x = scaled_x_dimension(inFile)
scaled_z = scaled_z_dimension(inFile)
print 'Original X is %ld' % inFile.X[0]
print 'ScaleX is %e' % inFile.header.scale[0]
print 'ScaleZ is %e' % inFile.header.scale[2]
print 'Offset is %f' % inFile.header.offset[0]
print 'Scaled X is %f' % scaled_x[0]
print inFile.X[1], inFile.X[2], inFile.X[3]
print scaled_x[1], scaled_x[2], scaled_x[3]
print inFile.Z[1],inFile.Z[2],inFile.Z[3]
print scaled_z[1],scaled_z[2],scaled_z[3]'''

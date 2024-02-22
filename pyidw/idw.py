import numpy as np
from math import sqrt, floor, ceil
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import fiona
from rasterio.enums import Resampling
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from matplotlib import colors


def show_map(input_raster='', colormap='coolwarm', image_size=1.5, return_figure=False):
    with rasterio.open(input_raster) as image_data:
        my_matrix = image_data.read(1)
        my_matrix = np.ma.masked_where(my_matrix == 32767, my_matrix)#32767不能改，为图像中的固定波段
        fig, ax = plt.subplots()
        image_hidden = ax.imshow(my_matrix, cmap=colormap)
        plt.close()

        fig, ax = plt.subplots()
        fig.set_facecolor("w")
        width = fig.get_size_inches()[0] * image_size
        height = fig.get_size_inches()[1] * image_size
        fig.set_size_inches(w=width, h=height)
        image = show(image_data, cmap=colormap, ax=ax)
        cbar = fig.colorbar(image_hidden, ax=ax, pad=0.02)
        if return_figure == False:
            plt.show()
        else:
            return fig, ax, cbar


#################################################


def crop_resize(input_raster_filename='',
                extent_shapefile_name='',
                max_height_or_width=250):
    # Here, co-variable raster file (elevation in this case) is croped and resized using rasterio.
    # 在这里，使用 rasterio 裁剪协变量栅格文件（在本例中为高程）并调整大小
    BD = gpd.read_file(extent_shapefile_name)
    elevation = rasterio.open(input_raster_filename)#背景文件

    # Using mask method from rasterio.mask to clip study area from larger elevation file.
    # 使用 rasterio.mask 中的 mask 方法从较大的高程文件中剪切研究区域
    croped_data, croped_transform = mask(dataset=elevation,
                                         shapes=BD.geometry,
                                         crop=True,
                                         all_touched=True)
    croped_meta = elevation.meta
    croped_meta.update({
        'height': croped_data.shape[-2],
        'width': croped_data.shape[-1],
        'transform': croped_transform
    })

    croped_filename = input_raster_filename.rsplit('.', 1)[0] + '_croped.tif'
    with rasterio.open(croped_filename, 'w', **croped_meta) as croped_file:
        croped_file.write(croped_data)  # Save the croped file as croped_elevation.tif to working directory.
        #将裁剪后的文件保存为croped_elevation.tif到工作目录

    # Calculate resampling factor for resizing the elevation file, this is done to reduce calculation time.
    # 计算重采样因子以调整高程文件的大小，这样做是为了减少计算时间
    # Here 250 is choosed for optimal result, it can be more or less depending on users desire.
    # max_height_or_width = 250
    resampling_factor = max_height_or_width / max(rasterio.open(croped_filename).shape)

    # Reshape/resize the croped elevation file and save it to working directory.
    # 调整裁剪后的高程文件的形状/大小并将其保存到工作目录resampled_elevation.tif
    with rasterio.open(croped_filename, 'r') as croped_elevation:

        resampled_elevation = croped_elevation.read(
            out_shape=(croped_elevation.count,
                       int(croped_elevation.height * resampling_factor),
                       int(croped_elevation.width * resampling_factor)),
            resampling=Resampling.bilinear)

        resampled_transform = croped_elevation.transform * croped_elevation.transform.scale(
            croped_elevation.width / resampled_elevation.shape[-1],
            croped_elevation.height / resampled_elevation.shape[-2])

        resampled_meta = croped_elevation.meta
        resampled_meta.update({
            'height': resampled_elevation.shape[-2],
            'width': resampled_elevation.shape[-1],
            'dtype': np.float64,
            'transform': resampled_transform
        })

        resampled_filename = input_raster_filename.rsplit(
            '.', 1)[0] + '_resized.tif'
        with rasterio.open(resampled_filename, 'w', **resampled_meta) as resampled_file:
            resampled_file.write(resampled_elevation )  # Save the resized file as resampled_elevation.tif in working directory.


#################################################


def blank_raster(extent_shapefile=''):
    calculationExtent = gpd.read_file(extent_shapefile)
    #提取所有坐标点中坐标最小与最大的点
    minX = floor(calculationExtent.bounds.minx)
    minY = floor(calculationExtent.bounds.miny)
    maxX = ceil(calculationExtent.bounds.maxx)
    maxY = ceil(calculationExtent.bounds.maxy)
    lonRange = sqrt((minX - maxX)**2)
    latRange = sqrt((minY - maxY)**2)

    #输入多少个点，输出的点数多少，最终点的运用比例是多少
    #输入193个点

    gridWidth = 10000
    # 网格宽度，初始值为400，数值越大，可视化效果越好
    # 面域面积越小，gridWidth的值要取得越大
    pixelPD = (gridWidth / lonRange)  # Pixel Per Degree每度像素
    gridHeight = floor(pixelPD * latRange)
    BlankGrid = np.ones([gridHeight, gridWidth])#np.ones()函数是NumPy中的一个函数，它用于创建指定形状的数组，并将数组中的元素初始化为1

    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'

    #通过得到筛分栅格的行数与列数、图像的宽度与长度，得到划分栅格的数量
    with rasterio.open(
            blank_filename,
            "w",# mode: 字符串(可选参数,默认是r，只读模式), 打开文件的模式，包括四种，分别是'r', 'w', 'r+', 'w+'，分别表示只读模式, 只写, 可读可写，可读可写模式。
            driver='GTiff',# driver：文件的格式，一般在r和r+模式会省略。若是创建文件则需要指定该参数。若是创建GeoTIFF文件，则需要指定driver=GTiff。
            height=BlankGrid.shape[0],# height：图像的高度，也即图像栅格矩阵的行数；
            width=BlankGrid.shape[1],# width：图像的宽度，也即图像栅格矩阵的列数；
            count=1,# count：图像的波段数；
            dtype=BlankGrid.dtype,  # BlankGrid.dtype, np.float32, np.int16
            crs=CRS.from_string(calculationExtent.crs.srs),
            transform=from_bounds(minX, minY, maxX, maxY, BlankGrid.shape[1], BlankGrid.shape[0]),
            # transform: 文件的仿射变换，可以是一个Affine对象或一个包含6个元素的列表或元组。
            # 这6个元素表示(x_size, skew_y, x_upper_left, skew_x, -y_size, y_upper_left), x为lon，y为lat.你可以自行计算出来，
            # 对于旋转系数skew_x和skew_x填写0.0即可，一般你不会使用到它的。由于在我们遥感影像上，坐标系的原点在左上角点，所以输入的角点信息也在左上角的经纬度坐标，
            # 由于是左上角点所以它计算是从左往右，从上往下，因此x上的分辨率正常，但是y上的分辨率需要添上负号因为从上往下纬度在减小。
            nodata=32767) as dst:# nodata：在矩阵中无效值的像素值，可以传入int，float，nan。
        dst.write(BlankGrid, 1)
        #rasterio.open函数用于创建一个DatasetReader或DatasetWriter对象，这两个对象分别用于读取和写入栅格数据.

#################################################

#获得IDW值
# def standard_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power, p_degree, s_radious):
def standard_idw(lon, lat, longs, lats, d_values, id_power, s_radious):
    """regression_idw is responsible for mathmatic calculation of idw interpolation with regression as a covariable."""
    # 回归idw负责以回归作为协变量进行idw插值的数学计算
    calc_arr = np.zeros(shape=(len(longs), 6))  # create an empty array shape of (total no. of observation * 6)/创建一个空数组形状（观察总数 * 6）
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.#第一列是已知数据点的经度
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.#第二列是已知数据点的纬度
    #     calc_arr[:, 2] = elevs    # Third column will be Elevation of known data points.第三列是已知数据点的高程，可忽略不计
    calc_arr[:, 3] = d_values  # Fourth column will be Observed data value of known data points.第四列是已知数据点的观测数据值
    calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:, 0] - lon) ** 2 + (calc_arr[:, 1] - lat) ** 2) ** id_power + 1)
    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)"第五列是 idw 公式“w = 1 / (d(x, x_i)^power + 1)”的权重值
    # >> constant 1 is to prevent int divide by zero when distance is zero.常量1是为了防止距离为零时int除以零

    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radious:, :]
    # Sort the array in ascendin order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radious "[ - s_radious :, :]"
    # 根据第 5 列（权重）“np.argsort(calc_arr[:,4])”对数组进行升序排序，并排除搜索radious "[ - s_radious :, :]" 之外的所有行

    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Sixth column is multiplicative product of inverse distant weight and actual value.#第六列是反距离权重与实际值的乘积

    idw1 = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.将加权值之和除以权重之和即可得到 IDW 插值
    #return idw1,calc_arr#求计算idw过程数据
    return idw1


#################################################


# 定义一个函数，用于进行反距离加权插值
def  idw_interpolation(input_point_shapefile='',  # 输入点的shapefile文件
                      extent_shapefile='',  # 范围的shapefile文件
                      column_name='',  # 列名
                      power=2,  # 幂
                      search_radious=4,  # 搜索半径
                      output_resolution=250):  # 输出分辨率
    blank_raster(extent_shapefile)  # 创建一个空白的栅格文件

    # 创建一个空白的文件名
    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'
    # 调整输入栅格文件的大小，使其与范围shapefile文件匹配，并设置最大高度或宽度
    crop_resize(input_raster_filename=blank_filename,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)

    # 创建一个调整大小后的栅格文件名
    resized_raster_name = blank_filename.rsplit('.', 1)[0] + '_resized.tif'
    # baseRasterFile = rasterio.open(resized_raster_name)
    # baseRasterFile stands for resampled elevation.#baseRasterFile 代表重新采样的高程

    # 使用rasterio库打开调整大小后的栅格文件
    with rasterio.open(resized_raster_name) as baseRasterFile:
        # 读取输入点的shapefile文件
        inputPoints = gpd.read_file(input_point_shapefile)
        # 创建一个空的数据框，用于存储观测数据
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        # obser_df代表observation_dataframe，lat，lon，每个点的data_value将存储在这里
        obser_df = pd.DataFrame()
        # 将输入点的第一列数据（站点名称）存储到数据框中
        obser_df['station_name'] = inputPoints.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        # 在高程栅格文件中创建两个站点经度、纬度索引列表
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],
            [lat for lat in inputPoints.geometry.y])
        obser_df['lon_index'] = lons  # 存储经度索引
        obser_df['lat_index'] = lats  # 存储纬度索引
        obser_df['data_value'] = inputPoints[column_name]  # 存储数据值

        # 读取基础栅格文件的第一层数据
        idw_array = baseRasterFile.read(1)
        # 遍历基础栅格文件的每一个像素
        for x in range(baseRasterFile.height):
            for y in range(baseRasterFile.width):
                # 如果像素值为32767，则跳过该像素
                if baseRasterFile.read(1)[x][y] == 32767:
                    continue
                else:
                    # 否则，使用标准的反距离加权插值方法计算该像素的值
                    idw_array[x][y] = standard_idw(
                        lon=x,
                        lat=y,
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        d_values=obser_df.data_value,
                        id_power=power,
                        s_radious=search_radious)

        # 创建输出文件名
        output_filename = input_point_shapefile.rsplit('.', 1)[0] + '_idw.tif'
        # 使用rasterio库打开输出文件，并写入插值后的数据
        with rasterio.open(output_filename, 'w', **baseRasterFile.meta) as std_idw:
            std_idw.write(idw_array, 1)

        # 显示地图
        show_map(output_filename)

#################################################


# 定义一个函数，用于计算标准IDW的准确性
def accuracy_standard_idw(input_point_shapefile='',  # 输入点的shapefile
                          extent_shapefile='',  # 范围的shapefile
                          column_name='',  # 列名
                          power=2,  # 幂
                          search_radious=4,  # 搜索半径
                          output_resolution=250):  # 输出分辨率
    blank_raster(extent_shapefile)  # 创建一个空白的栅格

    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'  # 创建一个空白的tif文件名
    crop_resize(input_raster_filename=blank_filename,  # 裁剪和调整大小
                extent_shapefile_name=extent_shapefile,  # 范围的shapefile名
                max_height_or_width=output_resolution)  # 最大高度或宽度

    resized_raster_name = blank_filename.rsplit('.', 1)[0] + '_resized.tif'  # 创建一个调整大小后的tif文件名

    with rasterio.open(resized_raster_name) as baseRasterFile:  # 打开调整大小后的tif文件
        inputPoints = gpd.read_file(input_point_shapefile)  # 读取输入点的shapefile
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        # obser_df代表observation_dataframe，lat，lon，每个站点的data_value将存储在这里。
        # obser_df代表观察数据框，每个站点的纬度、经度、数据值将在这里存储。
        obser_df = pd.DataFrame()  # 创建一个空的数据框
        obser_df['station_name'] = inputPoints.iloc[:, 0]  # 第一列是站点名称

        # create two list of indexes of station longitude, latitude in elevation raster file.
        # 在高程栅格文件中创建两个站点经度、纬度索引列表。
        # 在高程栅格文件中创建站点经度、纬度的索引列表。
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],  # 经度
            [lat for lat in inputPoints.geometry.y])  # 纬度
        obser_df['lon_index'] = lons  # 经度索引
        obser_df['lat_index'] = lats  # 纬度索引
        obser_df['data_value'] = inputPoints[column_name]  # 数据值
        obser_df['predicted'] = 0.0  # 预测值

        cv = LeaveOneOut()
        test_points_count = 0  # 初始化测试点的计数器
        for train_ix, test_ix in cv.split(obser_df):# 对数据框进行分割
            test_point = obser_df.iloc[test_ix[0]]# 测试点 #测试点数即为预测点数
            train_df = obser_df.iloc[train_ix]# 训练数据框

            obser_df.loc[test_ix[0], 'predicted'] = standard_idw(
                lon=test_point.lon_index,  # 测试点的经度
                lat=test_point.lon_index,  # 测试点的纬度
                longs=train_df.lon_index,  # 训练数据框的经度
                lats=train_df.lat_index,  # 训练数据框的纬度
                d_values=train_df.data_value,  # 训练数据框的数据值
                id_power=power,  # 幂
                s_radious=search_radious)  # 搜索半径
            test_points_count += 1  # 对每个测试点进行计数
        print(f"测试点的个数: {test_points_count}")  # 输出测试点的个数
        return obser_df.data_value.to_list(), obser_df.predicted.to_list(), test_points_count  # 返回数据值、预测值列表和测试点的个数


#################################################


def regression_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power,
                   p_degree, s_radious, x_max, x_min):
    """regression_idw is responsible for mathmatic calculation of idw interpolation with regression as a covariable."""
    calc_arr = np.zeros(shape=(len(longs), 6))  # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.
    calc_arr[:, 2] = elevs  # Third column will be Elevation of known data points.
    calc_arr[:, 3] = d_values  # Fourth column will be Observed data value of known data points.

    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)"
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:, 0] - lon) ** 2 + (calc_arr[:, 1] - lat) ** 2) ** id_power + 1)

    # Sort the array in ascendin order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radious "[ - s_radious :, :]"
    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radious:, :]

    # Sixth column is multiplicative product of inverse distant weight and actual value.
    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw1 = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()

    # Create polynomial regression equation where independent variable is elevation and dependent variable is data_value.
    # 创建多项式回归方程，其中自变量是海拔，因变量是 data_value。
    # Then, calculate R_squared value for just fitted polynomial equation.
    poly_reg = np.poly1d(np.polyfit(x=calc_arr[:, 2], y=calc_arr[:, 3], deg=p_degree))
    r_squared = r2_score(calc_arr[:, 3], poly_reg(calc_arr[:, 2]))

    regression_idw_combined = (1 - r_squared) * idw1 + r_squared * poly_reg(elev)
    if regression_idw_combined >= x_min and regression_idw_combined <= x_max:
        return regression_idw_combined
    elif regression_idw_combined < x_min:
        return x_min
    elif regression_idw_combined > x_max:
        return x_max


#################################################


class sigmoidStandardization():
    def __init__(self, input_array):
        self.in_array = input_array  # 输入数组
        self.arr_mean = self.in_array.mean()  # 数组的平均值
        self.arr_std = self.in_array.std()  # 数组的标准差

    def transform(self, number):
        # 对单个数字进行sigmoid标准化转换
        self.transformed = 1 / (1 + np.exp(-(number - self.arr_mean) / self.arr_std))
        return self.transformed

    def inverse_transform(self, number):
        # 对单个数字进行sigmoid标准化的逆转换
        self.reverse_transformed = np.log(number / (1 - number)) * self.arr_std + self.arr_mean
        return self.reverse_transformed


#################################################


def regression_idw_interpolation(input_point_shapefile='',
                                 input_raster_file='',
                                 extent_shapefile='',
                                 column_name='',
                                 power=2,
                                 polynomial_degree=1,
                                 search_radious=4,
                                 output_resolution=250):
    # 进行回归IDW插值
    crop_resize(input_raster_filename=input_raster_file,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)  # 裁剪和调整栅格文件的分辨率

    metStat = gpd.read_file(input_point_shapefile)  # 读取气象站点的shapefile文件

    resampled_filename = input_raster_file.rsplit('.', 1)[0] + '_resized.tif'  # 生成调整后的栅格文件名

    with rasterio.open(resampled_filename) as re_elevation:
        # 创建观测数据的DataFrame，包含站点的名称、经纬度索引、高程和数据值
        obser_df = pd.DataFrame()
        obser_df['station_name'] = metStat.iloc[:, 0]  # 站点名称

        # 创建站点经纬度在栅格文件中的索引列表
        lons, lats = re_elevation.index([lon for lon in metStat.geometry.x],
                                        [lat for lat in metStat.geometry.y])
        obser_df['lon_index'] = lons  # 经度索引
        obser_df['lat_index'] = lats  # 纬度索引
        obser_df['elevation'] = re_elevation.read(1)[lons, lats]  # 读取每个站点的高程数据
        obser_df['data_value'] = metStat[column_name]  # 读取每个站点的观测数据值
        obser_df['predicted'] = 0.0  # 初始化预测值列

        raster_transform = sigmoidStandardization(obser_df['elevation'])  # 对高程数据进行sigmoid标准化
        obser_df['trnsfrmd_raster'] = raster_transform.transform(obser_df['elevation'])  # 存储转换后的高程数据

        # 计算数据值的上下界范围
        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()

        regression_idw_array = re_elevation.read(1)  # 读取栅格数据
        for x in range(re_elevation.height):
            for y in range(re_elevation.width):
                # 对栅格中的每个像素进行处理
                if re_elevation.read(1)[x][y] == 32767:
                    continue  # 如果像素值为32767，则跳过
                else:
                    # 否则，使用回归IDW方法计算像素值
                    regression_idw_array[x][y] = regression_idw(
                        lon=x,
                        lat=y,
                        elev=raster_transform.transform(re_elevation.read(1)[x][y]),
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        elevs=obser_df['trnsfrmd_raster'],
                        d_values=obser_df.data_value,
                        id_power=power,
                        p_degree=polynomial_degree,
                        s_radious=search_radious,
                        x_max=upper_range,
                        x_min=lower_range)

    # 生成输出文件名
        output_filename = input_point_shapefile.rsplit('.', 1)[0] + '_regression_idw.tif'
        with rasterio.open(output_filename, 'w', **re_elevation.meta) as reg_idw:
            # 将计算后的回归IDW数组写入新的栅格文件
            reg_idw.write(regression_idw_array, 1)

        show_map(output_filename)  # 显示生成的栅格地图




#################################################


def accuracy_regression_idw(input_point_shapefile='',
                            input_raster_file='',
                            extent_shapefile='',
                            column_name='',
                            power=2,
                            polynomial_degree=1,
                            search_radious=4,
                            output_resolution=250):
    # 计算回归IDW插值的准确性
    crop_resize(input_raster_filename=input_raster_file,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)  # 裁剪和调整栅格文件的分辨率

    metStat = gpd.read_file(input_point_shapefile)  # 读取气象站点的shapefile文件

    resampled_filename = input_raster_file.rsplit('.', 1)[0] + '_resized.tif'  # 生成调整后的栅格文件名

    with rasterio.open(resampled_filename) as re_elevation:
        # 创建观测数据的DataFrame，包含站点的名称、经纬度索引、高程和数据值
        obser_df = pd.DataFrame()
        obser_df['station_name'] = metStat.iloc[:, 0]  # 站点名称

        # 创建站点经纬度在栅格文件中的索引列表
        lons, lats = re_elevation.index([lon for lon in metStat.geometry.x],
                                        [lat for lat in metStat.geometry.y])
        obser_df['lon_index'] = lons  # 经度索引
        obser_df['lat_index'] = lats  # 纬度索引
        obser_df['elevation'] = re_elevation.read(1)[lons, lats]  # 读取每个站点的高程数据
        obser_df['data_value'] = metStat[column_name]  # 读取每个站点的观测数据值
        obser_df['predicted'] = 0.0  # 初始化预测值列
        raster_transform = sigmoidStandardization(obser_df['elevation'])  # 对高程数据进行sigmoid标准化
        obser_df['trnsfrmd_raster'] = raster_transform.transform(obser_df['elevation'])  # 存储转换后的高程数据
        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()  # 计算数据值的上界
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()  # 计算数据值的下界

        cv = LeaveOneOut()  # 使用留一法交叉验证
        for train_ix, test_ix in cv.split(obser_df):
            # 对每个站点进行留一法验证
            test_point = obser_df.iloc[test_ix[0]]  # 测试点
            train_df = obser_df.iloc[train_ix]  # 训练数据集

            # 使用回归IDW方法对测试点进行预测
            obser_df.loc[test_ix[0], 'predicted'] = regression_idw(
                lon=test_point.lon_index,
                lat=test_point.lon_index,
                elev=test_point['trnsfrmd_raster'],
                longs=train_df.lon_index,
                lats=train_df.lat_index,
                elevs=train_df['trnsfrmd_raster'],
                d_values=train_df.data_value,
                id_power=power,
                p_degree=polynomial_degree,
                s_radious=search_radious,
                x_max=upper_range,
                x_min=lower_range)

        # 返回实际值和预测值的列表
        return obser_df.data_value.to_list(), obser_df.predicted.to_list()
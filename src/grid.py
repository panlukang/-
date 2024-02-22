# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


"""划分栅格"""

import math
import geopandas as gpd
from geopy.distance import distance
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPoint


geometry_field = 'geometry'
lon_field = 'lon'
lat_field = 'lat'


# 获取栅格预测点
def get_grid_data(polygon_gdf=None, meter_step=None, is_geo_coord=True):
    """
    切分面域，得到面域上结点的经纬度坐标
    :param polygon_gdf: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格区域大小, m
    :return: pd.Dataframe
    """

    geo_list = polygon_gdf[geometry_field].to_list()
    polygon_obj = unary_union(geo_list)

    # 根据栅格区域大小对面域进行栅格划分
    grid_gdf = generate_mesh(polygon_obj=polygon_obj, meter_step=meter_step, is_geo_coord=is_geo_coord)

    # 获取每个栅格中心点坐标
    grid_gdf[lon_field] = grid_gdf[geometry_field].apply(lambda x: x.centroid.x)
    grid_gdf[lat_field] = grid_gdf[geometry_field].apply(lambda x: x.centroid.y)

    return grid_gdf[['dx', 'dy', lon_field, lat_field, geometry_field]]


def generate_range(polygon_obj=None, meter_step=100, is_geo_coord=True):
    (min_x, min_y, max_x, max_y) = polygon_obj.bounds

    cen_x = polygon_obj.centroid.x
    cen_y = polygon_obj.centroid.y

    # 计算区域的长宽
    _width = max_y - min_y
    _length = max_x - min_x

    # 根据区域的中心点确定经纬度步长
    lon_step = get_geo_step(lon=cen_x, lat=cen_y, direction=1, step=meter_step, is_geo_coord=is_geo_coord)
    lat_step = get_geo_step(lon=cen_x, lat=cen_y, direction=0, step=meter_step, is_geo_coord=is_geo_coord)

    # 计算长宽多少个格子, 多生成一个, 做边界保护
    width_n = math.ceil(_width / lat_step) + 1
    length_n = math.ceil(_length / lon_step) + 1

    return lon_step, lat_step, width_n, length_n, min_x, max_y


# 逻辑子模块：生成栅格用于获取预测点
def generate_mesh(polygon_obj=None, meter_step=100, is_geo_coord=True):
    """
    生成栅格用于获取预测点
    :param polygon_obj: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格大小
    :return: gdf.GeoDataFrame
    """
    lon_step, lat_step, width_n, length_n, min_x, max_y = \
        generate_range(polygon_obj=polygon_obj, meter_step=meter_step, is_geo_coord=is_geo_coord)

    all_grid_list = []
    for n in range(width_n):
        point_list = [(min_x + k * lon_step, max_y - n * lat_step) for k in range(length_n)]

        def generate(xy):
            return Polygon([(xy[0], xy[1]), (xy[0] + lon_step, xy[1]),
                            (xy[0] + lon_step, xy[1] - lat_step), (xy[0], xy[1] - lat_step)])

        grid_list = list(map(generate, point_list))
        all_grid_list += grid_list

    index_list = [[i, j] for i in range(width_n) for j in range(length_n)]

    grid_gdf = gpd.GeoDataFrame({'mat_index': index_list}, geometry=all_grid_list, crs='EPSG:4326')

    # dx代表行索引, dy代表列索引
    grid_gdf['dx'] = grid_gdf['mat_index'].apply(lambda x: x[0])
    grid_gdf['dy'] = grid_gdf['mat_index'].apply(lambda x: x[1])
    grid_gdf.drop(columns='mat_index', axis=1, inplace=True)

    grid_gdf['bool'] = grid_gdf[geometry_field].apply(lambda x: x.intersects(polygon_obj))
    grid_gdf.drop(grid_gdf[grid_gdf['bool'] == False].index, axis=0, inplace=True)
    grid_gdf.reset_index(inplace=True, drop=True)
    grid_gdf.drop(columns='bool', inplace=True, axis=1)
    # res_grid_gdf = gpd.overlay(df1=polygon_gdf, df2=grid_gdf, how='intersection', keep_geom_type=True)
    return grid_gdf


def generate_range_by_point(point_list=None, meter_step=2000, is_geo_coord=True):

    # 依据point_df计算所有点的边界
    (min_x, min_y, max_x, max_y) = MultiPoint(point_list).bounds
    polygon_obj = Polygon([(min_x, min_y), (max_x, min_y),
                           (max_x, max_y), (min_x, max_y)])

    lon_step, lat_step, width_n, length_n, min_x, max_y = generate_range(polygon_obj=polygon_obj,
                                                                         meter_step=meter_step,
                                                                         is_geo_coord=is_geo_coord)

    return lon_step, lat_step, width_n, length_n, min_x, max_y


def get_row_col_index_by_loc(loc_x=None, loc_y=None,
                             region_min_x=None, region_max_y=None,
                             lon_step=None, lat_step=None):

    col_index = int((loc_x - region_min_x) / lon_step)
    row_index = int((region_max_y - loc_y) / lat_step)

    # print((loc_x, loc_y))
    # print((row_index, col_index))
    return (row_index, col_index)


# 逻辑子模块：确定经纬度步长
def get_geo_step(lon=None, lat=None, direction=1, step=100, is_geo_coord=True):
    """
    根据区域中心点确定经纬度步长
    :param lon: float, 经度
    :param lat: float, 纬度
    :param direction: int, 方向
    :param step: int, 步长
    :return:
    """

    if direction == 1:
        new_lon = lon + 0.1
        if is_geo_coord:
            dis = distance((lat, lon), (lat, new_lon)).m
        else:
            dis = 0.1
        return 0.1 / (dis / step)
    else:
        new_lat = lat + 0.1
        if is_geo_coord:
            dis = distance((lat, lon), (new_lat, lon)).m
        else:
            dis = 0.1
        return 0.1 / (dis / step)


if __name__ == '__main__':
    region = gpd.read_file(r'../data/input/bj/sy_region_gd.shp')
    region = region.to_crs('EPSG:32650')

    region['geometry'] = region['geometry'].apply(lambda x: x.buffer(1000))
    region = gpd.GeoDataFrame(region, crs='EPSG:32650')
    region = region.to_crs('EPSG:4326')

    grid = get_grid_data(polygon_gdf=region, meter_step=400, is_geo_coord=True)
    grid_gdf = gpd.GeoDataFrame(grid, geometry='geometry', crs='EPSG:4326')
    grid_gdf['grid_id'] = [i for i in range(1, len(grid) + 1)]
    grid_gdf.to_file(r'D:\trace_from_gd\data\input\bj\grid_400.shp', encoding='gbk', index=False)
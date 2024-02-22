import pandas as pd
import geopandas as gpd
from shapely import Point
from WebApi import GdRoutePlan
from pyidw import idw
from src.grid import get_grid_data

pd.set_option('expand_frame_repr', False)


def time_contour_generate(region_filepath: str,
                          grid_size: int,
                          ak: list,
                          destination: str,
                          grid_gdf_filepath: str,
                          points_gdf_filepath: str):
    """
    :param region_filepath: 等时圈生成范围shp路径
    :param grid_size: 划分格网尺寸
    :param ak: amap key
    :param destination: 目的地坐标例如'108.94733,34.34095'
    :param grid_gdf_filepath: 带cost属性的格网shp保存路径
    :param points_gdf_filepath: 带cost属性的格网中心点shp保存路径
    :return:
    """
    region_gdf = gpd.read_file(region_filepath)  # 输入

    # meter_step=500，单位m，改栅格尺寸
    grid_gdf = get_grid_data(polygon_gdf=region_gdf,
                             meter_step=grid_size, is_geo_coord=True)

    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry='geometry', crs='EPSG:4326')
    grid_gdf['grid_id'] = [i for i in range(1, len(grid_gdf) + 1)]

    # 根据生成的格网中心位置，请求路径数据，获取路径时间
    my_ak = ak
    my_plan = GdRoutePlan(key_list=my_ak)
    grid_gdf['cost'] = None

    for index, row in grid_gdf.iterrows():
        origin = str(row['lon']) + ',' + str(row['lat'])
        json_data, info_code = my_plan.walk_route_plan(origin=origin,
                                                       destination=destination,
                                                       alternative_route=1)
        _cost = json_data['route']['paths'][0]['cost']['duration']
        grid_gdf.loc[index, 'cost'] = int(_cost)

    grid_gdf.to_file(grid_gdf_filepath)
    points_gdf = grid_gdf.copy()
    points_gdf['geometry'] = points_gdf.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
    points_gdf.to_file(points_gdf_filepath)

    idw.idw_interpolation(
        input_point_shapefile=points_gdf_filepath,
        extent_shapefile=region_filepath,
        column_name="cost",
        power=2,  # 第四个参数power=是可选参数，默认值为2，这是idw方程中的功率参数。
        search_radious=10,  # 第五个参数search_radious搜索半径默认值为4，它决定了有多少个最近点将用于idw计算。
        output_resolution=250,  # 第六个参数 output_resolution默认值为 250。此参数定义生成的 _idw.tif 文件的最大高度或宽度（以像素为单位）。
    )  # 生成merge_points_idw.tif，生成的tif文件在你点文件的目录里


if __name__ == '__main__':

    time_contour_generate(region_filepath=r'./data/input/tmp.shp',
                          grid_size=300,
                          ak=[],
                          destination='108.94733,34.34095',
                          grid_gdf_filepath=r'./data/output/test_result2.shp',
                          points_gdf_filepath=r'./data/output/test_points2.shp')



# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


import requests
import pickle
import pandas as pd
import multiprocessing
import geopandas as gpd
from src.grid import get_grid_data
from shapely.geometry import LineString, Point

#生成栅格
def generate_grid():
    region_gdf = gpd.read_file(r'./data/input/长春市.shp')

    grid_gdf = get_grid_data(polygon_gdf=region_gdf,
                             meter_step=1000, is_geo_coord=True)

    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry='geometry', crs='EPSG:4326')
    grid_gdf['grid_id'] = [i for i in range(1, len(grid_gdf) + 1)]
    grid_gdf.to_file(r'./data/output/grid.shp', encoding='gbk')

#读取栅格
def from_to_match(grid_gdf=None):
    grid_info_df = pd.DataFrame()
    grid_info_df['grid_id'] = grid_gdf['grid_id']
    grid_info_df['left_bottom'] = grid_info_df['grid_id'].apply(lambda x: grid_gdf.loc[x, 'geometry'].bounds[:2])
    grid_info_df['right_top'] = grid_info_df['grid_id'].apply(lambda x: grid_gdf.loc[x, 'geometry'].bounds[2:])
    return grid_info_df

def add_from_to_grid(grid_gdf=None):
    # 生成包含所需信息的新表格
    grid_info_df = from_to_match(grid_gdf)
    grid_info_df.to_csv(r'./data/output/grid.csv', encoding='utf_8_sig', index=False)
    return grid_info_df







# 接口地址
url = "https://api.map.baidu.com/traffic/v1/bound"

# 此处填写你在控制台-应用管理-创建应用后获取的AK
ak = "QCvzkHCWxRYb2X8RFymtNthxmOHoifUL"

params = {
    "bounds": "39.912078,116.464303;39.918276,116.475442",
    "coord_type_input": "gcj02",
    "coord_type_output": "gcj02",
    "ak": 'MY0CKuv0wC2VtbS7IOvSzzQn4063LxcP',
    "road_grade": '1,2,3,4,5'

}

response = requests.get(url=url, params=params)
if response:
    print(response.json())

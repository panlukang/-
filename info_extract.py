# -*- coding: utf-8 -*-
# @Time : 2024/1/11 16:25
# @Author : Luke

import json
import warnings
import pandas as pd
import geopandas as gpd
from coord_trans import LngLatTransfer
from shapely.geometry import LineString, Point

warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)


def get_linestring_from_str(polyline_str: str) -> LineString:
    """
    根据坐标字符串返回LineString对象
    """
    coordinates_list = [tuple(map(float, pair.split(','))) for pair in polyline_str.split(';')]
    line = LineString(coordinates_list)
    return line


def get_path_from_segment(segment_dict: dict, path_id, link_id, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    从请求的公交规划json文件的segment结点，获取路径
    :param segment_dict:
    :param crs:
    :param path_id:
    :param link_id:
    :return:
    """
    path_gdf = gpd.GeoDataFrame(columns=['path_id', 'link_id', 'type', 'cost', 'geometry'], crs=crs)
    if 'walking' in segment_dict.keys():
        total_cost = float(segment_dict['walking']['cost']['duration'])
        total_distance = float(segment_dict['walking']['distance'])
        for step in segment_dict['walking']['steps']:
            polyline_str = step['polyline']['polyline']
            distance = float(step['distance'])
            cost = total_cost * distance / total_distance  # 请求的json文件未给出每段walk的时间cost，在此利用距离比值换算。
            tmp_line = get_linestring_from_str(polyline_str)
            dic = {'path_id': path_id,
                   'link_id': link_id,
                   'type': '步行',
                   'cost': float(cost),
                   'geometry': tmp_line}
            link_id += 1
            path_gdf = path_gdf._append(dic, ignore_index=True)
            del tmp_line, dic
    if 'bus' in segment_dict.keys():
        path_type = segment_dict['bus']['buslines'][0]['type']
        cost = segment_dict['bus']['buslines'][0]['cost']['duration']
        polyline_str = segment_dict['bus']['buslines'][0]['polyline']['polyline']  # 读取坐标字符串
        # 若存在进出口站点信息
        if 'entrance' in segment_dict['bus']['buslines'][0]['departure_stop'].keys():
            departure_stop = segment_dict['bus']['buslines'][0]['departure_stop']
            polyline_str = departure_stop['entrance']['location'] + ';' + polyline_str
        if 'exit' in segment_dict['bus']['buslines'][0]['arrival_stop'].keys():
            arrival_stop = segment_dict['bus']['buslines'][0]['arrival_stop']
            polyline_str = polyline_str + ';' + arrival_stop['exit']['location']
        tmp_line = get_linestring_from_str(polyline_str)
        dic = {'path_id': path_id,
               'link_id': link_id,
               'type': path_type,
               'cost': float(cost),
               'geometry': tmp_line}
        link_id += 1
        path_gdf = path_gdf._append(dic, ignore_index=True)
        del tmp_line, dic
    path_gdf.set_crs(crs=crs, inplace=True)
    return path_gdf


def get_path_from_bus_request_json(json_data, path_id=1, link_id=1, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    从请求的公交规划json文件，获取路径
    :param json_data:
    :param crs:
    :param path_id:
    :param link_id:
    :return:
    """
    path_gdf = gpd.GeoDataFrame(columns=['path_id', 'link_id', 'type', 'cost', 'geometry'], crs=crs)
    for transit in json_data['route']['transits']:
        for segment in transit['segments']:
            if len(path_gdf) == 0:
                path_gdf = path_gdf._append(get_path_from_segment(segment, path_id, link_id))
                link_id += 1
            else:
                link_id = path_gdf['link_id'].max() + 1
                path_gdf = path_gdf._append(get_path_from_segment(segment, path_id, link_id))
        path_id += 1
    path_gdf.reset_index(drop=True, inplace=True)
    return path_gdf


def get_path_from_walk_bike_request_json(json_data, type, path_id=1, link_id=1,
                                         crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
    """
    从请求的步行、骑行规划json文件，获取路径
    :param json_data:
    :param type:
    :param crs:
    :param path_id:
    :param link_id:
    :return:
    """
    path_gdf = gpd.GeoDataFrame(columns=['path_id', 'link_id', 'type', 'cost', 'road_name', 'geometry'], crs=crs)
    for path in json_data['route']['paths']:
        for step in path['steps']:
            cost = float(step['cost']['duration'])
            polyline_str = step['polyline']
            road_name = step['road_name']
            tmp_line = get_linestring_from_str(polyline_str)
            dic = {'path_id': path_id,
                   'link_id': link_id,
                   'type': type,
                   'cost': cost,
                   'road_name': road_name,
                   'geometry': tmp_line}
            link_id += 1
            path_gdf = path_gdf._append(dic, ignore_index=True)
        path_id += 1
    path_gdf.reset_index(drop=True, inplace=True)
    return path_gdf


def get_stop_info_dict(stop_dict: dict, stop_keys: str = None) -> dict:
    if stop_keys is None:
        stop_id = stop_dict['id']
        stop = Point(tuple(map(float, stop_dict['location'].split(','))))
        stop_name = stop_dict['name']
        dic = {'stop_id': stop_id,
               'stop_name': stop_name,
               'geometry': stop}
    else:
        stop_id = stop_dict['id']
        stop = Point(tuple(map(float, stop_dict[stop_keys]['location'].split(','))))
        stop_name = stop_dict['name']
        stop_entrance = stop_dict[stop_keys]['name']
        dic = {'stop_id': stop_id,
               'stop_name': stop_name,
               'stop_entrance': stop_entrance,
               'geometry': stop}
    return dic


def get_metro_stop_from_segment(segment_dict: dict, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame or None:
    """
    从请求的公交规划json文件的segment结点，获取地铁站点位置
    :param crs:
    :param segment_dict:
    :return:
    """
    if segment_dict['bus']['buslines'][0]['type'] != '地铁线路':
        return None
    metro_stop_gdf = gpd.GeoDataFrame(columns=['stop_id', 'stop_name', 'stop_entrance', 'geometry'], crs=crs)
    arrival_stop_dict = segment_dict['bus']['buslines'][0]['arrival_stop']
    departure_stop_dict = segment_dict['bus']['buslines'][0]['departure_stop']
    metro_stop_gdf = metro_stop_gdf._append(get_stop_info_dict(arrival_stop_dict), ignore_index=True)
    metro_stop_gdf = metro_stop_gdf._append(get_stop_info_dict(departure_stop_dict), ignore_index=True)
    if 'exit' in arrival_stop_dict.keys():
        dic = get_stop_info_dict(arrival_stop_dict, 'exit')
        metro_stop_gdf = metro_stop_gdf._append(dic, ignore_index=True)
    if 'entrance' in departure_stop_dict.keys():
        dic = get_stop_info_dict(departure_stop_dict, 'entrance')
        metro_stop_gdf = metro_stop_gdf._append(dic, ignore_index=True)
    metro_stop_gdf.set_crs(crs=crs, inplace=True)
    return metro_stop_gdf


def get_bus_stop_from_segment(segment_dict: dict, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame or None:
    """
    从请求的公交规划json文件的segment结点，获取公交站点位置
    :param crs:
    :param segment_dict:
    :return:
    """
    if segment_dict['bus']['buslines'][0]['type'] != '普通公交线路':
        return None
    bus_stop_gdf = gpd.GeoDataFrame(columns=['stop_id', 'bus_line', 'stop_name', 'geometry'], crs=crs)
    bus_line = segment_dict['bus']['buslines'][0]['name']
    arrival_stop_dict = segment_dict['bus']['buslines'][0]['arrival_stop']
    departure_stop_dict = segment_dict['bus']['buslines'][0]['departure_stop']
    dic1 = get_stop_info_dict(arrival_stop_dict)
    dic1['bus_line'] = bus_line
    dic2 = get_stop_info_dict(departure_stop_dict)
    dic2['bus_line'] = bus_line
    bus_stop_gdf = bus_stop_gdf._append(dic1, ignore_index=True)
    bus_stop_gdf = bus_stop_gdf._append(dic2, ignore_index=True)
    bus_stop_gdf.set_crs(crs=crs, inplace=True)
    return bus_stop_gdf


if __name__ == '__main__':
    pass
    # with open(r'./data/test_json.json', encoding="utf-8") as f:
    #     json_data = json.load(f)
    #
    # gdf = get_path_from_bus_request_json(json_data)
    # gdf.to_file('test1.geojson', driver="GeoJSON", encoding="utf-8")
    #
    # with open(r'test1.geojson', encoding="utf-8") as f:
    #     json_data = json.load(f)
    # json_data = add_timestamp_to_path(json_data, '2024-01-01 08:00:00')
    #
    # with open('test1.geojson', 'w') as f:
    #     json.dump(json_data, f, indent=2)

    # LngLatTransfer1 = LngLatTransfer()
    #
    # with open(r'walk_test.json', encoding="utf-8") as f:
    #     json_data = json.load(f)
    # gdf = get_path_from_walk_bike_request_json(json_data, '步行')
    # gdf['geometry'] = gdf.apply(lambda row: LngLatTransfer1.obj_convert(row['geometry'], 'gc-84'), axis=1)
    # gdf.to_file('walk_test.geojson', driver="GeoJSON", encoding="utf-8")
    # with open('walk_test.geojson', encoding="utf-8") as f:
    #     json_data = json.load(f)
    # json_data = add_timestamp_to_path(json_data, '2024-01-01 08:00:00')
    # with open('walk_test.geojson', 'w') as f:
    #     json.dump(json_data, f, indent=2)

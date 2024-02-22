# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


from shapely.geometry import LineString
import geopandas as gpd


walk_type_dict = {
    0: '普通道路', 1: '人行横道', 3: '地下通道', 4: '过街天桥', 5: '地铁通道',
    6: '公园', 7: '广场', 8: '扶梯', 9: '直梯', 10: '索道',
    11: '空中通道', 12: '建筑物穿越通道', 13: '行人通道', 14: '游船路线', 15: '观光车路线',
    16: '滑道', 18: '扩路', 19: '道路附属连接线', 20: '阶梯', 21: '斜坡', 22: '桥',
    23: '隧道', 30: '轮渡'}
def parse_gd_pt_route(json_data=None, parse_num=1):
    # 确定抽取的路径数(每个OD下)
    path_num = parse_num if parse_num <= len(json_data[0]['route']['transits']) else len(json_data[0]['route']['transits'])

    for i in range(0, path_num):
        parse_single_pt_route(json_data=json_data, seq=i)


def parse_single_pt_route(json_data=None, seq=1):
    # 确定抽取的路径数(每个OD下)

    route_item = json_data[0]['route']['transits'][seq]
    segments = route_item['segments']
    all_seq = 0
    for seg in segments:   # 每个方案包含数个seg
        for mode in seg.keys():  # 每个seg又包含不同模式的行程
            if mode == 'walking':
                mode_route_info = seg[mode] # 该模式的详细信息
                _mode_route_gdf = parse_walk_route(walk_route_info=mode_route_info)
                all_seq += 1
                _mode_route_gdf['all_seq'] = all_seq
            elif mode == 'bus':
                pass


def parse_walk_route(walk_route_info=None):
    try:
        walk_cost = walk_route_info['cost']['duration']
    except KeyError:
        walk_cost = 0

    # 解析途径路段信息
    road_name_list = []
    line_geo_list = []
    walk_type_list = []

    for step_item in walk_route_info['steps']:  # 每个模式的行程有line和cost
        try:
            road_name = step_item['road']
        except KeyError:
            road_name = '无名道路'

        line_geo = LineString(
            [list(map(float, coord_str.split(','))) for coord_str in step_item['polyline']['polyline'].split(';')])
        try:
            walk_type = walk_type_dict[int(step_item['navi']['walk_type'])]
        except KeyError:
            walk_type = '不确定'

        road_name_list.append(road_name)
        line_geo_list.append(line_geo)
        walk_type_list.append(walk_type)

    walk_route_gdf = gpd.GeoDataFrame({'road_name': road_name_list, 'walk_type': walk_type_list,
                                       'geometry': line_geo_list})
    walk_route_gdf['mode'] = 'walk'
    walk_route_gdf['cost'] = walk_cost
    walk_route_gdf['seq'] = [i for i in range(1, len(walk_route_gdf) + 1)]

    return walk_route_gdf


def parse_transit_route():
    pass
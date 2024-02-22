# -*- coding: utf-8 -*-
# @Time : 2024/1/11 16:25
# @Author : Luke

import json
import requests
import numpy as np


class GdRoutePlan(object):
    def __init__(self, key_list=None, ):
        self.key_list = key_list
        assert len(key_list) >= 1, '至少有一个key值'
        self.para_dict = {}
        self.api_url = None

    def reset_para_dict(self):
        key = self.key_list[np.random.randint(0, len(self.key_list))]
        self.para_dict = {'key': key}

    def request(self):
        print(self.para_dict)
        # 请求
        try:
            r = requests.get(self.api_url, params=self.para_dict, timeout=10)
            json_data = json.loads(r.text)
            info_code = json_data['infocode']
            print('请求成功')
            return json_data, int(info_code)
        except:
            print('请求失败')
            return None, None

    def car_route_plan(self, od_id=None, origin=None, destination=None,
                       origin_id=None, destination_id=None,
                       origin_type=None, avoidpolygons=None,
                       waypoints_loc=None, strategy='32', is_rnd_strategy=False):
        """
        # 参数含义见: https://lbs.amap.com/api/webservice/guide/api/newroute
        :param origin:
        :param destination:
        :param origin_id:
        :param destination_id:
        :param origin_type:
        :param avoidpolygons:
        :param od_id:
        :param waypoints_loc:
        :param strategy:
        :param is_rnd_strategy: 是否启用随机策略
        :return:
        """
        self.api_url = 'https://restapi.amap.com/v5/direction/driving'
        self.reset_para_dict()
        if is_rnd_strategy:
            strategy_list = ['0', '1', '2', '3', '32', '34', '35', '36', '37', '42']
            strategy = strategy_list[np.random.randint(0, len(strategy_list))]
        else:
            if strategy is None:
                strategy = '0'
        para_name = ['od_id', 'origin', 'destination',
                     'origin_id', 'destination_id',
                     'origin_type', 'avoidpolygons',
                     'waypoints_loc', 'strategy']
        para_val = [od_id, origin, destination,
                    origin_id, destination_id,
                    origin_type, avoidpolygons,
                    waypoints_loc, strategy]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                self.para_dict.update({name: val})
        self.para_dict.update({'show_fields': "cost,navi,tmcs,polyline"})
        # 请求
        json_data, info_code = self.request()

        return json_data, info_code

    def bus_route_plan(self, origin=None, destination=None,
                       origin_poi=None, destination_poi=None,
                       city1=None, city2=None, strategy='0',
                       alternative_route=2, multiexport=None,
                       max_trans=None, date=None, time=None):
        """
        # 参数含义见: https://lbs.amap.com/api/webservice/guide/api/newroute
        :param origin:起点经纬度 必填
        :param destination:目的地经纬度 必填
        :param origin_poi:起点POI ID
        :param destination_poi:目的地POI ID
        :param city1:起点所在城市 必填
        :param city2:目的地所在城市 必填
        :param strategy:公共交通换乘策略,默认高德推荐模式‘0’
        :param alternative_route:返回方案条数
        :param multiexport:返回地铁出入口数量
        :param max_trans:最大换乘次数
        :param date:
        :param time:
        :return:
        """
        self.api_url = 'https://restapi.amap.com/v5/direction/transit/integrated'
        self.reset_para_dict()

        para_name = ['origin', 'destination', 'originpoi',
                     'destinationpoi', 'city1', 'city2',
                     'strategy', 'AlternativeRoute', 'multiexport',
                     'max_trans', 'date', 'time']
        para_val = [origin, destination, origin_poi, destination_poi, city1, city2, strategy, alternative_route,
                    multiexport, max_trans, date, time]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                self.para_dict.update({name: val})
        self.para_dict.update({'show_fields': "cost,navi,polyline"})

        # 请求
        json_data, info_code = self.request()

        return json_data, info_code

    def walk_route_plan(self, origin=None, destination=None, alternative_route=3):
        """
        # 参数含义见: https://lbs.amap.com/api/webservice/guide/api/newroute
        :param origin: 起点经纬度 必填
        :param destination: 目的地经纬度 必填
        :param alternative_route: 返回方案条数
        :return:
        """
        self.api_url = 'https://restapi.amap.com/v5/direction/walking'
        self.reset_para_dict()

        para_name = ['origin', 'destination', 'alternative_route']
        para_val = [origin, destination, alternative_route]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                self.para_dict.update({name: val})
        self.para_dict.update({'show_fields': "cost,navi,polyline"})

        # 请求
        json_data, info_code = self.request()

        return json_data, info_code

    def bicycling_route_plan(self, origin=None, destination=None, alternative_route=3):
        """
        # 参数含义见: https://lbs.amap.com/api/webservice/guide/api/newroute
        :param origin: 起点经纬度 必填
        :param destination: 目的地经纬度 必填
        :param alternative_route: 返回方案条数
        :return:
        """
        self.api_url = 'https://restapi.amap.com/v5/direction/bicycling'
        self.reset_para_dict()

        para_name = ['origin', 'destination', 'alternative_route']
        para_val = [origin, destination, alternative_route]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                self.para_dict.update({name: val})
        self.para_dict.update({'show_fields': "cost,navi,polyline"})

        # 请求
        json_data, info_code = self.request()

        return json_data, info_code


class BdTrafficSituation(object):
    def __init__(self, ak_list=None):
        self.ak_list = ak_list
        assert len(ak_list) >= 1, '至少有一个ak值'

    def rectangle_situation(self, bounds=None, coord_type_input=None, id_label=None,
                            coord_type_output=None, road_grade=None):
        """
        参数含义: https://lbsyun.baidu.com/faq/api?title=webapi/traffic-rectangleseek
        :param bounds:
        :param coord_type_input:
        :param coord_type_output:
        :param road_grade:
        :param id_label: str, 标记参数
        :return:
        """
        # 接口地址
        url = "https://api.map.baidu.com/traffic/v1/bound"
        ak = self.ak_list[np.random.randint(0, len(self.ak_list))]
        para_dict = {'ak': ak}
        para_name = ['bounds', 'coord_type_input', 'coord_type_output', 'road_grade']
        para_val = [bounds, coord_type_input, coord_type_output, road_grade]
        for name, val in zip(para_name, para_val):
            if para_val is not None:
                para_dict.update({name: val})
        print(para_dict)
        try:
            r = requests.get(url, params=para_dict, timeout=10)
            json_data = r.json()
            info_code = json_data['status']
        except:
            return None, None
        # info_code == 1, 服务内部错误
        # info_code == 302, 天配额超限，限制访问
        return json_data, info_code


if __name__ == '__main__':
    my_ak = []
    my_plan = GdRoutePlan(key_list=my_ak)
    json_data, info_code = my_plan.walk_route_plan(origin='119.993353,30.289377', destination='120.00744,30.285542')
    with open('walk_test.json', 'w') as f:
        json.dump(json_data, f, indent=2)

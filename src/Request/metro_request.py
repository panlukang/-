# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


import json
import requests
from datetime import datetime


def request_metro_od_route(origin=None, destination=None, od_id=None,
                           key=None, max_trans=3, multiexport='1', alternative_route=3, strategy='0',
                           request_date=None, request_time=None,
                           o_city_code=None, d_city_code=None,):
    """

    :param origin: 起点经纬度, lng, lat, 经纬度小数点后不得超过6位
    :param destination: 终点经纬度, lng, lat, 经纬度小数点后不得超过6位
    :param od_id:
    :param key:
    :param request_date: 日期, 例如:2013-10-28
    :param request_time: 请求时间, 例如:9-54
    :param max_trans: 最大换乘次数, 0：直达, 1：最多换乘1次, 2：最多换乘2次, 3：最多换乘3次, 4：最多换乘4次
    :param multiexport: 地铁出入口数量, 0：只返回一个地铁出入口, 1：返回全部地铁出入口
    :param alternative_route: 返回方案条数, 可传入1-10的阿拉伯数字，代表返回的不同条数
    :param strategy: 可选值：
                    0：推荐模式，综合权重，同高德APP默认
                    1：最经济模式，票价最低
                    2：最少换乘模式，换乘次数少
                    3：最少步行模式，尽可能减少步行距离
                    4：最舒适模式，尽可能乘坐空调车
                    5：不乘地铁模式，不乘坐地铁路线
                    6：地铁图模式，起终点都是地铁站
                    （地铁图模式下originpoi及destinationpoi为必填项）
                    7：地铁优先模式，步行距离不超过4KM
                    8：时间短模式，方案花费总时间最少
    :return:
    """
    now_date = datetime.now()
    if request_date is None:
        request_date = rf'{now_date.year}-{now_date.month}-{now_date.day}'
    if request_time is None:
        request_time = rf'{now_date.hour}-{now_date.minute}'

    api_url = 'https://restapi.amap.com/v5/direction/transit/integrated'

    para_dict = {
        'key': key,
        'show_fields': "cost,navi,polyline",
        'city1': o_city_code, 'city2': d_city_code,
        'origin': origin, 'destination': destination,
        'strategy': str(strategy), 'date': str(request_date),
        'time': str(request_time),
        'AlternativeRoute': str(alternative_route), 'multiexport': str(multiexport),
        'max_trans': str(max_trans),
    }
    print(para_dict)

    try:
        r = requests.get(api_url, params=para_dict, timeout=10)
        json_data = json.loads(r.text)
        info_code = json_data['infocode']
    except:
        print(rf"{od_id}请求失败..., timeout...")
        return None, None

    info_code_inf_dict = {"10000": 'OK', '10001': 'Key不正确', '10003': '日配额超限',
                          '10004': '单位时间内访问过于频繁'}

    if info_code == "10000":
        print(rf"{od_id}请求成功...")
        return json_data, info_code
    elif info_code == '10003':
        print(rf"{od_id}请求失败..., info_code: {info_code}")
        return None, info_code
    else:
        print(rf"{od_id}请求失败..., info_code: {info_code}")
        return None, info_code




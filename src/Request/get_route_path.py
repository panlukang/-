# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


from src.Request.metro_request import request_metro_od_route
from src.Parse.gd_pt_route_path import parse_gd_pt_route


if __name__ == '__main__':
    json_res = request_metro_od_route(origin=r'109.040891,34.371156',
                                      destination=r'108.947133,34.269918',
                                      od_id=1,
                                      key='02a0764c920b2b248ca71e29bf673db9',
                                      max_trans=3, multiexport='1', alternative_route=3, strategy='7',
                                      o_city_code='029', d_city_code='029',
                                      request_date=None, request_time=None)
    parse_gd_pt_route(json_data=json_res, parse_num=5)
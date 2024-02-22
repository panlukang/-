# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


"""将全量OD的地铁出行部分转化为地铁的进出站数据"""

import pandas as pd
import geopandas as gpd
import multiprocessing
from shapely.geometry import Point


def main_match(metro_df=None, od_id_field='od_id', o_x_field=None, o_y_field=None,  d_x_field=None, d_y_field=None,
               stop_gdf=None,  stop_id_field='stop_id', slice_num=10, initial_buffer=1000, buffer_gap=400,
               max_buffer=50000, utm_crs='EPSG:32649', path_cost_dict=None):
    """
    将全量od数据的起终点进行配站, 并且计算起点/终点与站点的直线距离(m)
    :param metro_df:
    :param od_id_field:
    :param o_x_field:
    :param o_y_field:
    :param d_x_field:
    :param d_y_field:
    :param stop_gdf:
    :param stop_id_field:
    :param slice_num:
    :param initial_buffer:
    :param buffer_gap:
    :param max_buffer:
    :param utm_crs:
    :param path_cost_dict: {(o_stop, d_stop): time_cost}
    :return:
    """
    # 数据分成N个slice
    slice_list = cut_slice_for_df(df=metro_df, n=slice_num)
    del metro_df

    # 每个切片并行
    pool = multiprocessing.Pool(slice_num)
    res_list = []
    for _metro_df in slice_list:
        res = pool.apply_async(format_subway_od, args=(stop_gdf, _metro_df[[od_id_field, o_x_field, o_y_field,
                                                                            d_x_field, d_y_field]],
                                                       o_x_field, o_y_field, d_x_field, d_y_field, stop_id_field,
                                                       od_id_field, initial_buffer, buffer_gap, max_buffer, utm_crs,
                                                       path_cost_dict))

        res_list.append(res)

    pool.close()
    pool.join()

    od_match_res_df = pd.DataFrame()
    for res in res_list:
        od_match_res_df = pd.concat([od_match_res_df, res.get()])

    od_match_res_df.reset_index(inplace=True, drop=True)
    return od_match_res_df


def format_subway_od(stop_gdf=None, gd_all_od_df=None,
                     o_x_field=None, o_y_field=None, d_x_field=None, d_y_field=None, stop_id_field=None,
                     od_id_field=None, initial_buffer=100, buffer_gap=400, max_buffer=50000, utm_crs='EPSG:32649',
                     path_cost_dict=None):
    """

    :param stop_gdf:
    :param gd_all_od_df:
    :param o_x_field:
    :param o_y_field:
    :param d_x_field:
    :param d_y_field:
    :param stop_id_field:
    :param od_id_field:
    :param initial_buffer:
    :param utm_crs:
    :param buffer_gap:
    :param max_buffer:
    :param path_cost_dict: {(o_stop, d_stop): cost}
    :return:
    """
    gd_all_od_df['from_p'] = gd_all_od_df[[o_x_field, o_y_field]].apply(lambda x: Point(x), axis=1)
    gd_all_od_df['to_p'] = gd_all_od_df[[d_x_field, d_y_field]].apply(lambda x: Point(x), axis=1)

    gd_all_od_gdf = gpd.GeoDataFrame(gd_all_od_df, geometry='from_p', crs='EPSG:4326')
    del gd_all_od_df
    gd_all_od_gdf = gd_all_od_gdf.to_crs(utm_crs)

    gd_all_od_gdf.set_geometry('to_p', crs='EPSG:4326', inplace=True)
    gd_all_od_gdf = gd_all_od_gdf.to_crs(utm_crs)

    stop_gdf = stop_gdf.to_crs(utm_crs)

    # od_id_field, 'entrance_stop_id', 'l'
    o_match_res = increment_match(od_gdf=gd_all_od_gdf, od_id_field=od_id_field,
                                  stop_id_field=stop_id_field,
                                  od_geo_field='from_p', stop_gdf=stop_gdf,
                                  match_id_field='entrance_stop_id',
                                  initial_buffer=initial_buffer, buffer_gap=buffer_gap, max_buffer=max_buffer)

    # od_id_field, 'exit_stop_id', 'l'
    d_match_res = increment_match(od_gdf=gd_all_od_gdf, od_id_field=od_id_field,
                                  stop_id_field=stop_id_field,
                                  od_geo_field='to_p', stop_gdf=stop_gdf,
                                  match_id_field='exit_stop_id',
                                  initial_buffer=initial_buffer, buffer_gap=buffer_gap, max_buffer=max_buffer)

    # od_id, entrance_stop_id, exit_stop_id, l_x, l_y
    # 1, [1, 2, 3], [11, 22, 21], [12.1, 13.1], [33.1, 12.1]
    od_match_res = pd.merge(o_match_res, d_match_res, on=od_id_field)

    del o_match_res
    del d_match_res
    del gd_all_od_gdf

    od_match_res['final_match'] = od_match_res[['entrance_stop_id', 'exit_stop_id',
                                                'l_x', 'l_y']].apply(
        lambda item: get_minimum_se(entrance_stop_list=item[0],
                                    exit_stop_list=item[1],
                                    l_to_entrance_list=item[2],
                                    l_to_exit_list=item[3],
                                    path_cost_dict=path_cost_dict), axis=1)

    od_match_res['entrance_stop_id'] = od_match_res['final_match'].apply(lambda x: x[0])
    od_match_res['exit_stop_id'] = od_match_res['final_match'].apply(lambda x: x[1])
    od_match_res['l_x'] = od_match_res['final_match'].apply(lambda x: x[2])
    od_match_res['l_y'] = od_match_res['final_match'].apply(lambda x: x[3])
    od_match_res.drop(columns=['final_match'], axis=1, inplace=True)
    return od_match_res


def get_minimum_se(entrance_stop_list=None, exit_stop_list=None,
                   l_to_entrance_list=None, l_to_exit_list=None, path_cost_dict=None):
    """

    :param entrance_stop_list:
    :param exit_stop_list:
    :param l_to_entrance_list:
    :param l_to_exit_list:
    :param path_cost_dict:
    :return:
    """
    walk_speed = 1.8 # m/s
    pt_speed = 10  # m/s
    final_entrance_stop, final_exit_stop, final_l_to_entrance, final_l_to_exit = None, None, None, None

    if len(entrance_stop_list) == 1 and len(exit_stop_list) == 1:
        return entrance_stop_list[0], exit_stop_list[0], l_to_entrance_list[0], l_to_exit_list[0]
    else:
        minimum_cost = 9999999
        for entrance_stop, l_to_entrance in zip(entrance_stop_list, l_to_entrance_list):
            for exit_stop, l_to_exit in zip(exit_stop_list, l_to_exit_list):
                connection_time_a = (l_to_entrance / walk_speed if l_to_entrance <= 1500 else l_to_entrance / pt_speed) / 60 # min
                connection_time_b = (l_to_exit / walk_speed if l_to_exit <= 1500 else l_to_exit / pt_speed) / 60 # min

                if exit_stop == entrance_stop:
                    now_time_cost = connection_time_b + connection_time_a
                else:
                    now_time_cost = path_cost_dict[(entrance_stop, exit_stop)] + connection_time_b + connection_time_a

                if now_time_cost <= minimum_cost:
                    minimum_cost = now_time_cost
                    final_entrance_stop, final_exit_stop = entrance_stop, exit_stop
                    final_l_to_entrance = l_to_entrance
                    final_l_to_exit = l_to_exit

        return final_entrance_stop, final_exit_stop, final_l_to_entrance, final_l_to_exit


def increment_match(od_gdf=None, od_id_field='od_id', od_geo_field=None, stop_gdf=None, stop_id_field=None,
                    initial_buffer=1000, buffer_gap=400, max_buffer=50000,
                    match_id_field='entrance_stop_id'):

    od_gdf.set_geometry(od_geo_field, crs=od_gdf.crs, inplace=True)

    remain_od_list = od_gdf[od_id_field].to_list()
    origin_od_num = len(remain_od_list)
    all_done_od_gdf = gpd.GeoDataFrame()

    for gap in [i for i in range(0, max_buffer, buffer_gap)]:
        print(rf'buffer: {initial_buffer + gap}m...')
        od_gdf = od_gdf[od_gdf[od_id_field].isin(remain_od_list)]
        od_gdf.reset_index(inplace=True, drop=True)
        stop_gdf['stop_buffer'] = stop_gdf['geometry'].apply(lambda p: p.buffer(initial_buffer + gap))
        stop_gdf.set_geometry('stop_buffer', inplace=True, crs=od_gdf.crs)

        join_gdf = gpd.sjoin(od_gdf, stop_gdf, how='left')
        join_gdf.reset_index(inplace=True, drop=True)

        done_od_gdf = join_gdf[~join_gdf[stop_id_field].isna()]
        all_done_od_gdf = pd.concat([all_done_od_gdf, done_od_gdf])

        remain_od_list = list(join_gdf[join_gdf[stop_id_field].isna()][od_id_field].unique())

        if not remain_od_list:
            break
    print(rf'还剩下{len(remain_od_list)}个OD未曾关联到...')
    print(rf'原来{origin_od_num}个od, 关联后{len(all_done_od_gdf)}条信息')
    all_done_od_gdf[stop_id_field] = all_done_od_gdf[stop_id_field].astype(int)
    all_done_od_gdf.reset_index(inplace=True, drop=True)
    all_done_od_gdf['l'] = all_done_od_gdf[[od_geo_field, 'geometry']].apply(lambda item: item[0].distance(item[1]),
                                                                             axis=1)

    # all_done_od_gdf.sort_values(by=[od_id_field, 'l'], ascending=[True, True], inplace=True)
    # match_res = all_done_od_gdf.groupby([od_id_field]).first().reset_index(drop=False)

    match_res = all_done_od_gdf.groupby([od_id_field]).agg({stop_id_field: list, 'l': list}).reset_index(drop=False)

    match_res.rename(columns={stop_id_field: match_id_field}, inplace=True)
    return match_res[[od_id_field, match_id_field, 'l']]


def cut_slice_for_df(df=None, n=10):
    df['__id'] = [i for i in range(0, len(df))]
    df['slice'] = pd.cut(df['__id'], bins=n, labels=[i for i in range(0, n)])
    df_list = [df[df['slice'] == i] for i in range(0, n)]
    df.drop(columns=['__id', 'slice'], axis=1, inplace=True)

    return df_list


if __name__ == '__main__':
    pass

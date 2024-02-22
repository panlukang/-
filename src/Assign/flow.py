# -- coding: utf-8 --
# @Time    : 2023/12/4 16:22
# @Author  : TangKai
# @Team    : ZheChengData

"""流量统计模块"""


import pandas as pd
import networkx as nx
from src.globalVal import GlobalVal

glb = GlobalVal()


def main_flow(od_df=None, origins_field=None, destination_field=None, path_df=None,
          from_node_field='from_node', to_node_field='to_node', stop_id_name_dict=None,
          o_time_field=None, path_name='link_path', demand_field=None):

    """
    流量计算主模块
    :param od_df: pd.DataFrame(), 地铁od, 字段: {origins_field}, {destinations_field}, {o_time_field}, demand_field_list,
    :param origins_field: str, od起点站点ID字段
    :param destination_field: str, od终点站点ID字段
    :param path_df: pd.DataFrame(),
    :param from_node_field:
    :param to_node_field:
    :param stop_id_name_dict:
    :param o_time_field:
    :param path_name:
    :param demand_field:
    :return:
    """

    # 全网客流统计
    all_flow_df = all_flow(od_df=od_df, path_df=path_df,
                           from_stop_field=from_node_field, to_stop_field=to_node_field,
                           origins_field=origins_field, destinations_field=destination_field,
                           demand_field=demand_field, o_time_field=o_time_field)

    # 进站客流统计
    in_stop_df = stop_in_out_flow(od_df=od_df, stop_id_name_dict=stop_id_name_dict, origins_field=origins_field,
                                  demand_field=demand_field, o_time_field=o_time_field)

    section_flow_df = section_flow(od_df=od_df, path_df=path_df, origins_field=origins_field,
                                   destination_field=destination_field, o_time_field=o_time_field,
                                   from_node_field=from_node_field, to_node_field=to_node_field,
                                   path_name=path_name, demand_field=demand_field)

    line_flow_df = line_flow(od_df=od_df, path_df=path_df, origins_field=origins_field,
                             destination_field=destination_field,
                             from_node_field=from_node_field, to_node_field=to_node_field,
                             demand_field=demand_field, o_time_field=o_time_field)

    return all_flow_df, in_stop_df, section_flow_df, line_flow_df


def all_flow(od_df=None, path_df=None,
             from_stop_field=None,to_stop_field=None,
             origins_field=None, destinations_field=None,
             demand_field=None, o_time_field=None):
    """
    分时段的全网进站\换乘\总客流计算
    :param od_df:
    :param path_df:
    :param from_stop_field:
    :param to_stop_field:
    :param origins_field:
    :param destinations_field:
    :param demand_field: str, OD量字段
    :param o_time_field:
    :return: pd.DataFrame(), {o_time_field}, 进站量, 换乘量, 总客流

    """
    # 和总出行次数, 进站量
    print(r'线网总客流计算......')

    # 分时段的进站总量(出行)
    all_travel_times_df = od_df.groupby([o_time_field])[[demand_field]].sum().reset_index(drop=False)

    # 地铁网总客流(进站量 + 换乘量)
    od_df = pd.merge(od_df, path_df, left_on=[origins_field, destinations_field],
                     right_on=[from_stop_field, to_stop_field])

    od_df[[demand_field]] = od_df[[demand_field]].multiply(od_df["ratio"], axis="index")

    # 换乘量
    od_df['xfer_num'] = od_df['xfer_list'].apply(lambda x: len(x))
    od_df['xfer_num'] = od_df['xfer_num'] * od_df['ratio'] * od_df['uv']

    # 分时段的换乘总量
    all_xfer_times_df = od_df.groupby([o_time_field])[['xfer_num']].sum().reset_index(drop=False)

    all_flow_df = pd.merge(all_travel_times_df, all_xfer_times_df, on=o_time_field)
    all_flow_df.fillna(0, inplace=True)
    all_flow_df['总客流'] = all_flow_df[demand_field] + all_flow_df['xfer_num']
    all_flow_df.rename(columns={'uv': '进站量',  'xfer_num': '换乘量'}, inplace=True)
    all_flow_df.set_index(o_time_field, inplace=True)
    return all_flow_df


def stop_in_out_flow(od_df=None, stop_id_name_dict=None,origins_field=None,
                     demand_field=None, o_time_field=None):
    """
    分时段的站点进站客流计算
    :param od_df:
    :param stop_id_name_dict:
    :param origins_field:
    :param demand_field:
    :param o_time_field:
    :return:
    """
    print(r'进站客流计算......')
    in_stop_df = od_df.groupby([o_time_field, origins_field])[demand_field].sum().reset_index(drop=False)
    in_stop_df.rename(columns={demand_field: '进站量'}, inplace=True)

    in_stop_df['站点名称'] = in_stop_df[origins_field].apply(lambda x: stop_id_name_dict[x])
    in_stop_df.drop(columns=[origins_field], inplace=True, axis=1)
    in_stop_df.set_index([o_time_field, '站点名称'], inplace=True)

    return in_stop_df


def section_flow(od_df=None, path_df=None, origins_field=None, destination_field=None,
                 o_time_field=None, from_node_field=None, to_node_field=None,
                 path_name=None, demand_field=None):
    """
    分时段的断面客流计算
    :param od_df:
    :param path_df:
    :param origins_field:
    :param destination_field:
    :param o_time_field:
    :param from_node_field:
    :param to_node_field:
    :param path_name:
    :param demand_field:
    :return:
    """
    # 断面客流
    print(r'断面客流计算......')
    # 将路径数据连接到od表上
    od_demand_path_df = pd.merge(od_df, path_df,
                                 left_on=[origins_field, destination_field],
                                 right_on=[from_node_field, to_node_field])

    # 基于od_demand_path_df做全有全无分配
    od_demand_path_df[[demand_field]] = od_demand_path_df[[demand_field]].multiply(od_demand_path_df["ratio"],
                                                                                   axis="index")

    # 断面客流
    od_demand_path_df = od_demand_path_df.explode(column=path_name, ignore_index=True)
    flow_df = od_demand_path_df.groupby([path_name, o_time_field])[[demand_field]].sum().reset_index(drop=False)
    del od_demand_path_df
    flow_df[from_node_field] = flow_df[path_name].apply(lambda x: int(x[0]))
    flow_df[to_node_field] = flow_df[path_name].apply(lambda x: int(x[1]))
    flow_df.drop(path_name, inplace=True, axis=1)

    return flow_df


def line_flow(path_df=None, od_df=None, origins_field=None, destination_field=None, demand_field=None,
              from_node_field=None, to_node_field=None, o_time_field=None):
    """

    :param path_df:
    :param od_df:
    :param origins_field:
    :param destination_field:
    :param demand_field:
    :param from_node_field:
    :param to_node_field:
    :param o_time_field:
    :return:
    """
    print(r'线路客流计算...')
    od_demand_path_df = pd.merge(od_df, path_df,
                                 left_on=[origins_field, destination_field],
                                 right_on=[from_node_field, to_node_field])

    od_demand_path_df[[demand_field]] = od_demand_path_df[[demand_field]].multiply(od_demand_path_df["ratio"],
                                                                                         axis="index")
    od_demand_path_df = od_demand_path_df[[demand_field] + [o_time_field, 'line_list']].copy()
    od_demand_path_df = od_demand_path_df.explode(column=['line_list'], ignore_index=True)

    line_flow_df = od_demand_path_df.groupby(['line_list',
                                              o_time_field])[[demand_field]].sum().reset_index(drop=False).rename(
        columns={'line_list': 'line_name'})
    line_flow_df.set_index([o_time_field, 'line_name'], inplace=True)
    return line_flow_df


if __name__ == '__main__':
    # print(drop_dup(val_list=[1, 1, 1, 2, 2, 1, 1, 2, 2, 2]))

    g = nx.DiGraph()
    g.add_edges_from([(1,2, {'w':1}), (2,3, {'w': 12})])

    for i, k in zip(nx.all_shortest_paths(g, source=1, target=3, weight='w'), [x for x in range(3)]):
        print(i)
        print(k)

    # for i in nx.all_simple_paths(g, source=1, target=3):
    #     print(i)
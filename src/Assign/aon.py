# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


"""地铁全有全无分配"""

import pandas as pd
import networkx as nx
from src.globalVal import GlobalVal
from src.Assign.net import create_graph


glb = GlobalVal()


def all_or_nothing(from_node_field='from_node', to_node_field='to_node', route_df=None, weight_field=None,
                   p_ft_line_dict=None, is_k_path=False):

    """
    计算所有的站点之间的最短路
    :param from_node_field: str, 线网的上游站点字段名称
    :param to_node_field: str, 线网的上游站点字段名称
    :param route_df: pd.DataFrame(), 线网
    :param weight_field: str, 权重搜路字段
    :param p_ft_line_dict: dict, {(p_from, p_to): line_name}
    :param is_k_path: bool, 是否启用k路径搜路
    :return: pd.DataFrame(), {from_name}, {to_name}, link_path, time_cost, seq, ratio, line_list, xfer_list
    """

    # 依据地铁网络建立图
    graph_net = create_graph(link_gdf=route_df,
                             from_node_field=from_node_field, to_node_field=to_node_field,
                             weight_field_list=[weight_field])
    if is_k_path:
        # 尚未完善
        path_df = pd.DataFrame()
    else:
        # 搜索路径
        path_df = get_path(net_graph=graph_net, weight_name=weight_field,
                           from_name=from_node_field, to_name=to_node_field)

    # 计算line_id列
    path_df['line_list'] = path_df['link_path'].apply(lambda link_path: [p_ft_line_dict[link] for link in link_path])

    # 计算换乘节点[(stop, from_line, to_line), ...]
    path_df['xfer_list'] = path_df[['link_path', 'line_list']].apply(lambda x: calc_xfer(link_path_list=x[0],
                                                                                       line_path_list=x[1]), axis=1)
    # 计算途径的线路名称
    path_df['line_list'] = path_df['line_list'].apply(
        lambda line_path: drop_dup(line_path))
    return path_df


def get_path(net_graph=None, weight_name=None, node_id_list=None, from_name=None, to_name=None):
    """
    计算所有节点之间最短路径link序列, 一个OD起终点只返回一条最短路径
    :param net_graph:
    :param weight_name:
    :param node_id_list:
    :param from_name:
    :param to_name:
    :return: pd.DataFrame(), 字段: {from_name}, {to_name}, link_path, time_cost, seq, ratio
    """

    # 所有节点
    all_node_list = list(net_graph.nodes)
    if node_id_list is None or not all_node_list:
        node_id_list = all_node_list.copy()

    # 非形心结点
    non_taz_id_list = list(set(all_node_list) - set(node_id_list))

    # 计算每个小区到其他所有节点的最短路径, 这里还是有冗余计算, nx好像不支持两个节点集合之间的最短路径
    all_path = {node_id: dict(nx.multi_source_dijkstra_path(net_graph, {node_id}, weight=weight_name)) for node_id in
                node_id_list}

    path_data = pd.DataFrame(all_path)
    path_data.drop(non_taz_id_list, inplace=True, axis=0)
    stack_series = path_data.stack()

    stack_data = pd.DataFrame(stack_series, columns=['node_path'])
    stack_data.reset_index(drop=False, inplace=True)
    stack_data.rename(columns={'level_0': to_name, 'level_1': from_name}, inplace=True)

    stack_data['link_path'] = stack_data['node_path'].apply(lambda x: [(x[i], x[i + 1]) for i in range(0, len(x) - 1)])
    # 去除同站
    stack_data.drop(stack_data[stack_data[from_name] == stack_data[to_name]].index, inplace=True, axis=0)
    path_df = stack_data.reset_index(drop=True)
    path_df.drop(columns=['node_path'], axis=1, inplace=True)

    # 获取线路的总开销
    p_from_to_cost_dict = nx.get_edge_attributes(net_graph, weight_name)
    path_df['time_cost'] = path_df['link_path'].apply(
        lambda link_path: [p_from_to_cost_dict[item] for item in link_path])
    path_df['seq'] = path_df['link_path'].apply(lambda x: [i for i in range(1, len(x) + 1)])
    path_df['ratio'] = 1

    return path_df


def calc_xfer(link_path_list=None, line_path_list=None):
    """
    计算换乘信息
    :param link_path_list:
    :param line_path_list:
    :return:
    """
    xfer_list = []
    for i in range(0, len(line_path_list) - 1):
        if line_path_list[i] != line_path_list[i + 1]:
            xfer_list.append([(link_path_list[i][-1], line_path_list[i], line_path_list[i + 1])])
        else:
            pass
    return xfer_list


def drop_dup(val_list=None):
    """
    不改变元素顺序去重
    :param val_list:
    :return:
    """
    res_list = list(set(val_list))
    res_list.sort(key=val_list.index)
    return res_list



# 待开发
def get_k_path(net_graph=None, weight_name=None, node_id_list=None, from_name=None, to_name=None):
    """
    计算所有小区之间最短路径link序列, 一个OD起终点只返回K条最短路径
    :param net_graph:
    :param weight_name:
    :param node_id_list:
    :param from_name:
    :param to_name:
    :return:
    """

    # 所有节点
    all_node_list = list(net_graph.nodes)
    if node_id_list is None or not all_node_list:
        node_id_list = all_node_list.copy()
    # 非形心(小区)结点
    non_taz_id_list = list(set(all_node_list) - set(node_id_list))

    od_pair_list = [(o, d) for o in node_id_list for d in node_id_list if o != d]

    nx.all_simple_paths()
    all_path = {node_id: dict(nx.multi_source_dijkstra_path(net_graph, {node_id}, weight=weight_name)) for node_id in node_id_list}

    path_data = pd.DataFrame(all_path)
    path_data.drop(non_taz_id_list, inplace=True, axis=0)
    stack_series = path_data.stack()

    stack_data = pd.DataFrame(stack_series, columns=['node_path'])
    stack_data.reset_index(drop=False, inplace=True)
    stack_data.rename(columns={'level_0': to_name, 'level_1': from_name}, inplace=True)

    stack_data['link_path'] = stack_data['node_path'].apply(lambda x: [(x[i], x[i+1]) for i in range(0, len(x) - 1)])
    # 去除同站
    stack_data.drop(stack_data[stack_data[from_name] == stack_data[to_name]].index, inplace=True, axis=0)
    path_df = stack_data.reset_index(drop=True)
    path_df.drop(columns=['node_path'], axis=1, inplace=True)

    # 获取线路的总开销
    p_from_to_cost_dict = nx.get_edge_attributes(net_graph, weight_name)

    path_df['time_cost'] = path_df['link_path'].apply(lambda link_path: [p_from_to_cost_dict[item] for item in link_path])
    path_df['seq'] = path_df['link_path'].apply(lambda x: [i for i in range(1, len(x) + 1)])
    path_df['ratio'] = 1

    return path_df


# 待开发
def k_path(od_pair_list=None, g=None, weight_field=None, k=3, path_id_start=1):
    path_list = []
    path_id_list = []
    od_pair_df = pd.DataFrame({'od_pair': []})
    od_pair_df['od_pair'] = od_pair_list
    od_pair_df['od_path'] = od_pair_df['od_pair'].apply(
        lambda od_pair: [(item, i) for item, i in zip(nx.all_shortest_paths(g, source=od_pair[0],
                                                                            target=od_pair[1], weight=weight_field),
                                                      [x for x in range(0, k)])])


if __name__ == '__main__':
    # print(drop_dup(val_list=[1, 1, 1, 2, 2, 1, 1, 2, 2, 2]))

    g = nx.DiGraph()
    g.add_edges_from([(1,2, {'w':1}), (2,3, {'w': 12})])

    for i, k in zip(nx.all_shortest_paths(g, source=1, target=3, weight='w'), [x for x in range(3)]):
        print(i)
        print(k)

    # for i in nx.all_simple_paths(g, source=1, target=3):
    #     print(i)
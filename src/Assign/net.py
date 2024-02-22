# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData

"""构网模块"""

import networkx as nx


# 依据路网建立networkx图对象-主函数
def create_graph(link_gdf=None, from_node_field=None, to_node_field=None,
                 weight_field_list=None):
    """
    建立图文件
    :param link_gdf:
    :param from_node_field:
    :param to_node_field:
    :param weight_field_list:
    :return:
    """

    # 取出计算所需要的字段
    skim_link = link_gdf[[from_node_field, to_node_field] + weight_field_list].copy()

    # 建立有向图对象
    net_graph = nx.DiGraph()

    # 获得边列表
    edge_list = get_edge_list(df=skim_link,
                              from_node_field=from_node_field,
                              to_node_field=to_node_field,
                              weight_field_list=weight_field_list)
    net_graph.add_edges_from(edge_list)

    return net_graph


# 依据路网建立networkx图对象-子函数2
def get_edge_list(df=None, from_node_field=None, to_node_field=None, weight_field_list=None):
    """
    生成边列表用于创建图
    :param df: pd.DataFrame, 路网数据
    :param from_node_field: str, 起始节点字段名称
    :param to_node_field: str, 起始节点字段名称
    :param weight_field_list: list, 代表边权重的字段列表名称
    :return:
    """

    # 起终点
    from_list = [from_node for from_node in df[from_node_field].to_list()]
    to_list = [to_node for to_node in df[to_node_field].to_list()]

    if weight_field_list is not None:
        # 这一步非常重要, 保证迭代的顺序是按照用户传入的列顺序
        weight_data = df[weight_field_list].copy()

        # 获取权重字典
        weight_list = [list(item) for item in weight_data.itertuples(index=False)]

        # 边列表
        edge_list = [[from_node, to_node, dict(zip(weight_field_list, data))]
                     for from_node, to_node, data in zip(from_list, to_list, weight_list)]
    else:
        # 边列表
        edge_list = [[from_node, to_node] for from_node, to_node in zip(from_list, to_list)]

    return edge_list


# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData

"""标准化轨道网络图层"""


import pandas as pd
import networkx as nx
from shapely.geometry import LineString



def generate_net(regenerate_logical=True, route_gdf=None,
                 route_id_field=None, _route_id_field=None, time_field=None,
                 line_name_field=None, from_stop_field=None, to_stop_field=None, route_speed_dict=None,
                 stop_gdf=None, phy_stop_id_field=None, stop_name_field=None, year_field=None, search_year=2022,
                 frequency_field=None, utm_crs=None,
                 time_period_list=None, capacity_field=None):

    # 首先是构建全部年份的图层
    print(stop_gdf.columns)
    # 先检查站点的物理id是否重复
    assert len(stop_gdf[phy_stop_id_field].unique()) == len(stop_gdf[phy_stop_id_field]), '站点的物理id有重复!'

    # 检查route层用到的点是否都在stop层中
    route_used_stop_set = set(list(route_gdf[from_stop_field].unique()) + list(route_gdf[to_stop_field].unique()))
    stop_used_stop_set = set(stop_gdf[phy_stop_id_field].unique())
    if route_used_stop_set - stop_used_stop_set:

        print('route层中用到的stop数量多于stop层!')
    elif stop_used_stop_set - route_used_stop_set:
        print(stop_used_stop_set - route_used_stop_set)
        print('stop层中用到的stop数量多于route层!')
    else:
        pass

    # 生成物理站点-> 站点名称的映射
    p_stop_id_name_dict = {p_stop_id: stop_name for p_stop_id, stop_name in zip(stop_gdf[phy_stop_id_field],
                                                                                stop_gdf[stop_name_field])}
    # 生成每条route的站序
    # 同时分配逻辑站点, 注意一条线的上下行的同一物理站点不区分逻辑站点
    route_id_list, stop_list, stop_index_list, logical_stop_id_list = [], [], [], []
    logical_start = 1
    route_line_id_dict = {}
    route_line_name_dict = {}
    line_id = 1
    route_name_dict = {}
    route_se_stop_dict = {}
    reverse_line_dict = {}

    for routes_id, route_link in route_gdf.groupby([route_id_field, _route_id_field]):

        reverse_line_dict[routes_id[0]] = routes_id[1]

        route_link.reset_index(inplace=True, drop=True)

        # route到line_name和line_id的映射
        route_line_id_dict[routes_id[0]] = line_id
        route_line_id_dict[routes_id[1]] = line_id

        route_line_name_dict[routes_id[0]] = route_link.at[0, line_name_field]
        route_line_name_dict[routes_id[1]] = route_link.at[0, line_name_field]

        line_id += 1

        # 直接按照from_to建立无向图
        g = nx.Graph()
        edge_list = get_edge_list(df=route_link, from_node_field=from_stop_field, to_node_field=to_stop_field)
        g.add_edges_from(edge_list)

        # 找到起始站点
        start_stop_id, is_loop = find_first_stop(g=g, route_link=route_link,
                                                 from_stop_field=from_stop_field, process_routes_id=routes_id)

        # 站点序列
        stop_seq = list(nx.dfs_preorder_nodes(g, start_stop_id))
        print(stop_seq)
        # 正向站序
        route_id_list.append(routes_id[0])
        stop_list.append(stop_seq)
        route_name_dict[routes_id[0]] = route_link.at[0, line_name_field] + \
                                        ':' + p_stop_id_name_dict[stop_seq[0]] + '-' + p_stop_id_name_dict[stop_seq[-1]]

        route_se_stop_dict[routes_id[0]] = (stop_seq[0], stop_seq[-1])
        route_se_stop_dict[routes_id[1]] = (stop_seq[-1], stop_seq[0])

        # 反向站序
        route_id_list.append(routes_id[1])
        stop_list.append(stop_seq[::-1])
        route_name_dict[routes_id[1]] = route_link.at[0, line_name_field] + \
                                        ':' + p_stop_id_name_dict[stop_seq[-1]] + '-' + p_stop_id_name_dict[stop_seq[0]]

        # 站点index
        stop_index_list.append([i for i in range(1, len(stop_seq) + 1)])
        stop_index_list.append([i for i in range(1, len(stop_seq) + 1)])

        # 逻辑站点
        temp_logical_id_list = [logical_start + i for i in range(0, len(stop_seq))]
        logical_stop_id_list.append(temp_logical_id_list)
        logical_stop_id_list.append(temp_logical_id_list[::-1])

        # 更新
        logical_start += len(stop_seq)

    # 线路站点的关联表
    route_stop_df = pd.DataFrame({'route_id': route_id_list,
                                  'stop_id': stop_list,
                                  'stop_index': stop_index_list,
                                  'l_stop_id': logical_stop_id_list})

    route_stop_df = route_stop_df.explode(column=['stop_id', 'stop_index', 'l_stop_id'], ignore_index=True)

    # 逻辑站点到line_id的映射
    logical_line_id_dict = {l: route_line_id_dict[route_id]
                            for l, route_id in zip(route_stop_df['l_stop_id'], route_stop_df['route_id'])}

    # 逻辑站点到物理站点的映射
    logical_phy_dict = {l: p for l, p in zip(route_stop_df['l_stop_id'], route_stop_df['stop_id'])}

    # 建立(route_id, phy_stop): logical_stop
    routeStop_logical_dict = {(int(route_id), int(phy_id)): int(l_id) for route_id, phy_id, l_id in
                              zip(route_stop_df['route_id'],
                                  route_stop_df['stop_id'],
                                  route_stop_df['l_stop_id'])}

    # 物理站点到逻辑站点的映射, 一个物理站点应该会对应多个逻辑站点, 这里tag到任意一个, 用于转换矩阵的行和列
    phy_logical_dict = {p: l for l, p in logical_phy_dict.items()}

    # 标记正向route
    route_gdf['type'] = 'route_pos'

    # 地铁的反向route
    reverse_route_gdf = route_gdf.copy()
    reverse_route_gdf['type'] = 'route_neg'
    reverse_route_gdf[route_id_field] = reverse_route_gdf[_route_id_field]
    reverse_route_gdf[[from_stop_field, to_stop_field]] = route_gdf[[to_stop_field, from_stop_field]]
    reverse_route_gdf['geometry'] = reverse_route_gdf['geometry'].apply(lambda x: LineString(list(x.coords)[::-1]))

    route_gdf = pd.concat([route_gdf, reverse_route_gdf])
    route_gdf.reset_index(inplace=True, drop=True)
    route_gdf = route_gdf.to_crs(utm_crs)
    route_gdf['length'] = route_gdf['geometry'].apply(lambda x: x.length / 1000)
    route_gdf = route_gdf.to_crs('EPSG:4326')

    # 字段规范化
    stop_gdf = stop_gdf[[phy_stop_id_field, stop_name_field, 'geometry']].copy()
    stop_gdf.rename(columns={phy_stop_id_field: 'stop_id', stop_name_field: 'stop_name'}, inplace=True)

    # 建立route到capacity\frequency的映射关系
    route_cap_fre_df = route_gdf[
        [route_id_field, capacity_field] + [period + rf'_{frequency_field}' for period in time_period_list]].copy()
    route_cap_fre_df.rename(
        columns={period + rf'_{frequency_field}': period.lower() + '_frequency' for period in time_period_list},
        inplace=True)
    route_cap_fre_df.drop_duplicates(subset=[route_id_field], inplace=True, keep='first')
    route_cap_fre_df.rename(columns={capacity_field: 'capacity'}, inplace=True)
    route_cap_fre_df.set_index(route_id_field, inplace=True)

    # route_id, p_from, p_to, time, length, type, geometry
    route_gdf = route_gdf[
        [route_id_field, from_stop_field, to_stop_field, time_field, line_name_field,
         year_field, 'type', 'length', 'geometry']].copy()
    route_gdf.rename(columns={route_id_field: 'route_id',
                              from_stop_field: 'p_from', to_stop_field: 'p_to', time_field: 'time',
                              line_name_field: 'line_name', year_field: 'year'}, inplace=True)

    # route_id, p_from, p_to, l_from, l_to, time, length, type, 'line_name', geometry
    route_gdf['l_from'] = route_gdf[['route_id', 'p_from']].apply(lambda x: routeStop_logical_dict[(x[0], x[1])], axis=1)
    route_gdf['l_to'] = route_gdf[['route_id', 'p_to']].apply(lambda x: routeStop_logical_dict[(x[0], x[1])], axis=1)

    # 添加逻辑上的换乘连杆
    # l_from, l_to, p_from, p_to, time, type, length
    transit_link_df = add_transit_link(route_stop_df=route_stop_df,
                                       logical_phy_dict=logical_phy_dict)
    transit_link_df['type'] = 'transit'
    transit_link_df['year'] = -1900
    transit_link_df['route_id'] = -999
    transit_link_df['line_name'] = 'transfer'
    transit_link_df['route_name'] = 'transfer'

    # 用于搜路的df
    search_net_df = pd.concat([route_gdf, transit_link_df])
    search_net_df.reset_index(inplace=True, drop=True)

    # section_id
    search_net_df['section_id'] = [i for i in range(1, len(search_net_df) + 1)]

    # 用于展示的网络
    show_route_gdf = search_net_df.dropna(subset=['geometry'])

    # 记录换乘连杆的section_id
    transit_section_list = search_net_df[search_net_df['type'] == 'transit']['section_id'].to_list()

    # 逻辑from-to到section的映射
    l_ft_section_dict = {(f, t): route_id for f, t, route_id in zip(search_net_df['l_from'],
                                                                    search_net_df['l_to'],
                                                                    search_net_df['section_id'])}

    # section编号到route的映射
    section_route_id_dict = {section_id: route_id for section_id, route_id in zip(search_net_df['section_id'],
                                                                                  search_net_df['route_id'])}

    # 逻辑站点到route的映射
    logical_route_dict = {logical: route for logical, route in zip(route_stop_df['l_stop_id'],
                                                                   route_stop_df['route_id'])}

    # 断面id到断面名称的映射
    section_name_dict = {section_id: line_name + ':' + p_stop_id_name_dict[f] + '-' + p_stop_id_name_dict[t]
                         for section_id, f, t, line_name in zip(search_net_df['section_id'],
                                                                search_net_df['p_from'],
                                                                search_net_df['p_to'],
                                                                search_net_df['line_name'])}

    # 依据年份筛选route和连杆
    used_route_df = search_net_df[(search_net_df['year'] <= search_year) &
                                    (search_net_df['type'].isin(['route_neg', 'route_pos']))].copy()
    used_route_df.reset_index(inplace=True, drop=True)
    used_phy_stop_list = list(set(list(used_route_df['p_from'].unique()) + list(used_route_df['p_to'].unique())))

    used_transit_df = search_net_df[(search_net_df['type'] == 'transit') &
                                    (search_net_df['p_from'].isin(used_phy_stop_list))].copy()
    search_net_df = pd.concat([used_route_df, used_transit_df])
    search_net_df.reset_index(inplace=True, drop=True)

    # 物理from-to到section的映射(存在两条不同的线路经过同一区间的情况, 所以需要from, to, route)
    # (from, to, route) -> section
    # 这一步要放在筛选年份后做
    r_p_ft_section_dict = {}
    # 找出并线区间, (f, t): [r1, r2]以及(route, from, to) -> section
    for _, row in search_net_df.iterrows():
        f, t, r, section_id = row['p_from'], row['p_to'], row['route_id'], row['section_id']
        if f != t:
            r_p_ft_section_dict[(r, f, t)] = section_id

    # check #
    # 因为这个search_net_df是由完整的网络通过年份筛选而来的, 如果原来的完整网络的断面年份标错了, 可能导致一条route上的断面不连续
    # 即这里要确保经过筛选后的网络, 在同一条route上只能有且只有一条连续的线或者一个环
    check_route(search_net_df=search_net_df, from_field='p_from', to_field='p_to', route_field='route_id')
    # check #

    # 基于站点-线路关联表计算换乘站点
    stop_count_df = pd.DataFrame(route_stop_df['stop_id'].value_counts()).reset_index(drop=False)
    xfer_stop_list = stop_count_df[stop_count_df['count'] > 2]['stop_id'].to_list()

    return search_net_df, show_route_gdf, stop_gdf, route_stop_df, transit_section_list, \
           r_p_ft_section_dict, l_ft_section_dict, logical_route_dict, logical_phy_dict, phy_logical_dict, \
           p_stop_id_name_dict, route_line_id_dict, logical_line_id_dict, section_route_id_dict, route_line_name_dict, \
           route_name_dict, xfer_stop_list, route_se_stop_dict, reverse_line_dict, route_cap_fre_df, section_name_dict


# 逻辑子模块: 生成边列表用于创建图
def get_edge_list(df=None, from_node_field=None, to_node_field=None, weight_field_list=None):
    """
    生成边列表用于创建图
    :param df: pd.DataFrame, 路网数据
    :param from_node_field: str, 起始节点字段名称
    :param to_node_field: str, 终到节点字段名称
    :param weight_field_list: list, 代表边权重的字段列表名称
    :return: edge_list
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
        edge_list = [(from_node, to_node, dict(zip(weight_field_list, data)))
                     for from_node, to_node, data in zip(from_list, to_list, weight_list)]
    else:
        # 边列表
        edge_list = [(from_node, to_node) for from_node, to_node in zip(from_list, to_list)]

    return edge_list


def check_route(search_net_df=None, from_field=None, to_field=None, route_field=None):
    net_df = search_net_df.copy()
    for r, route_df in net_df.groupby(route_field):
        g = nx.Graph()
        edge_list = get_edge_list(df=route_df,
                                  from_node_field=from_field, to_node_field=to_field)
        g.add_edges_from(edge_list)
        degree_dict = dict(nx.degree(g))

        # 找出度为1的节点
        one_degree_stop_list = [stop for stop in list(degree_dict.keys()) if degree_dict[stop] == 1]
        if not one_degree_stop_list:
            if set(degree_dict.values()) != {2}:
                raise ValueError(rf'route:{r}线路有问题...')
        elif len(one_degree_stop_list) != 2:
            raise ValueError(rf'route:{r}线路有问题...')


def find_first_stop(g=None, route_link=None, from_stop_field=None, process_routes_id=None):
    """
    记得考虑环线
    :param g:
    :param route_link:
    :param from_stop_field:
    :param process_routes_id:
    :return:
    """
    degree_dict = dict(nx.degree(g))
    is_loop = False

    # 找出度为1的节点
    one_degree_stop_list = [stop for stop in list(degree_dict.keys()) if degree_dict[stop] == 1]
    # print(one_degree_stop_list)
    start_stop = -99
    # 如果没有度为1的点则是环
    if not one_degree_stop_list:
        # 进一步检查是否是环, 环的所有节点的度都是2
        if set(degree_dict.values()) != {2}:
            raise ValueError(f'route{process_routes_id}路径存在问题!')
        # 任取一个点作为开始点
        start_stop = route_link.at[0, from_stop_field]
        is_loop = True
    else:
        if len(one_degree_stop_list) <= 1:
            raise ValueError(f'route{process_routes_id}路径存在问题!')
        elif len(one_degree_stop_list) > 2:
            raise ValueError(f'route{process_routes_id}路径存在问题!')
        else:
            # 肯定有一个stop处于from列
            from_stop_list = route_link[from_stop_field].to_list()
            for candidate_stop in one_degree_stop_list:
                if candidate_stop in from_stop_list:
                    start_stop = candidate_stop
                    break
                else:
                    pass
        if start_stop == -99:
            raise ValueError(f'route{process_routes_id}路径存在问题!')
    return start_stop, is_loop


# 添加连杆
def add_transit_link(route_stop_df=None, logical_phy_dict=None):

    transit_link_df = pd.DataFrame()

    # 按照物理站点聚合, 先对逻辑站点去重, 排除掉
    for _, _df in route_stop_df.groupby('stop_id'):
        logical_list = _df['l_stop_id'].to_list()
        transit_ft = [(f, t) for f in logical_list for t in logical_list if f != t]
        transit_df = pd.DataFrame(transit_ft, columns=['l_from', 'l_to'])
        transit_link_df = pd.concat([transit_link_df, transit_df])

    transit_link_df.reset_index(inplace=True, drop=True)
    transit_link_df['time'] = 0.5

    transit_link_df['p_from'] = transit_link_df['l_from'].map(logical_phy_dict)
    transit_link_df['p_to'] = transit_link_df['l_to'].map(logical_phy_dict)
    transit_link_df['length'] = 0.20
    transit_link_df['type'] = 'transit'
    # l_from, l_to, p_from, p_to, type, length
    return transit_link_df



if __name__ == '__main__':

    g = nx.Graph()
    g.add_edges_from([(1, 2), (2, 3), (3, 1)])
    print(list(nx.dfs_preorder_nodes(g, 2)))



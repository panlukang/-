# -- coding: utf-8 --
# @Time    : 2023/11/27 11:00
# @Author  : TangKai
# @Team    : ZheChengData


"""
some global val
"""


class GlobalVal():
    def __init__(self):

        self.FROM_NODE_FIELD = 'from_node'
        self.TO_NODE_FIELD = 'to_node'
        self.DIRECTION_FIELD = 'dir'
        self.LINK_ID_FIELD = 'link_id'
        self.NODE_ID_FIELD = 'node_id'
        self.GEOMETRY_FIELD = 'geometry'
        self.IS_CENTROIDS_FIELD = 'is_centroids'



if __name__ == '__main__':
    obj = GlobalVal()
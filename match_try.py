import networkx as nx
import numpy as np
import os
import cv2


def mea2box(mea):
    center_x = mea[0] / 2
    center_y = mea[1] / 2
    w = mea[2]
    h = mea[3]
    return [int(i) for i in [center_x - w / 2, center_y - h / 2, center_x + w / 2, center_y + h / 2]]


def box2mea(box):
    x = box[0]
    y = box[1]
    w = box[2] - x
    h = box[3] - y
    return [x, y, w, h]


def cal_iou(state, measure):  # 求交并比
    state = mea2box(state)  # [lt_x, lt_y, rb_x, rb_y].T
    measure = mea2box(measure)  # [lt_x, lt_y, rb_x, rb_y].T
    s_tl_x, s_tl_y, s_br_x, s_br_y = state[0], state[1], state[2], state[3]
    m_tl_x, m_tl_y, m_br_x, m_br_y = measure[0], measure[1], measure[2], measure[3]
    # 计算相交部分的坐标
    x_min = max(s_tl_x, m_tl_x)
    x_max = min(s_br_x, m_br_x)
    y_min = max(s_tl_y, m_tl_y)
    y_max = min(s_br_y, m_br_y)
    inter_h = max(y_max - y_min + 1, 0)
    inter_w = max(x_max - x_min + 1, 0)
    inter = inter_h * inter_w
    if inter == 0:
        return 0
    else:
        return inter / ((s_br_x - s_tl_x) * (s_br_y - s_tl_y) + (m_br_x - m_tl_x) * (m_br_y - m_tl_y) - inter)


def match(state_list, measure_list):
    graph = nx.Graph()  # 创建无向图
    for state_index, state in enumerate(state_list):  # 遍历预测值列表
        state_node = 'state_%d' % state_index  # 获取预测值序号
        graph.add_node(state_node, bipartite=0)  # 添加节点到0
        for measure_index, measure in enumerate(measure_list):  # 遍历量测值列表
            mea_node = 'mea_%d' % measure_index  # 获取量测值序号
            graph.add_node(mea_node, bipartite=1)  # 添加节点到1
            score = cal_iou(state, measure)  # 计算交并比
            if score is not None:
                graph.add_edge(state_node, mea_node, weight=score)  # 将交并比设定为权值
    match_set = nx.max_weight_matching(graph)  # 找到权重最大
    res = dict()
    for (node_1, node_2) in match_set:
        if node_1.split('_')[0] == 'mea':
            node_1, node_2 = node_2, node_1  # 换成统一位置
        res[node_1] = node_2
    return res

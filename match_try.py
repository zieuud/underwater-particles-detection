import networkx as nx


class Box:
    def __init__(self, location):
        self.location = location
        self.box_track = [location]
        xx, yy, ww, hh = location
        self.track = [(xx + hh // 2, yy + ww // 2)]
        self.terminate_count = 0

    def location_add(self, new_location):
        self.location = new_location
        self.box_track.append(new_location)
        xx, yy, ww, hh = new_location
        self.track.append((xx + ww // 2, yy + hh // 2))

    def not_found(self):
        self.terminate_count += 1
        self.location_add(self.location)


def progressive_scan_identi(gray_input):
    for row in range(len(gray_input)):
        gray_scale = gray_input[row]
        outliers = []
        base = gray_scale[0]
        for column in range(len(gray_scale)):
            if column < len(gray_scale) - 1 and int(gray_scale[column + 1]) - int(base) > 5:
                outliers.append(column + 1)
            else:
                base = gray_scale[column + 1] if column < len(gray_scale) - 1 else 0
        for k in outliers:
            gray_input[row, k] = 255
    gray_temp = gray_input.T
    for row in range(len(gray_temp)):
        gray_scale = gray_temp[row]
        white = 0
        gap = 0
        for column in range(len(gray_scale)):
            if gray_scale[column] == 255:
                if 0 < gap < 3:
                    gray_scale[column - 1] = 255
                    gray_scale[column - 2] = 255
                white += 1
                gap = 0
            else:
                gray_scale[column] = 0
                if white == 1:
                    gray_scale[column - 1] = 0
                gap += 1
                white = 0
    gray_output = gray_temp.T
    return gray_output


def box2mea(box):
    x, y, w, h = box
    x0 = x + w
    y0 = y + h
    return [x, y, x0, y0]


def cal_iou(state, measure):  # 求交并比
    state = box2mea(state)  # [lt_x, lt_y, rb_x, rb_y].T
    measure = box2mea(measure)  # [lt_x, lt_y, rb_x, rb_y].T
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

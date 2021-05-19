import numpy as np


# 计算欧式距离
def cal_distance(point1, point2):
    dis = np.sqrt(np.sum(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1])))
    return dis


# 基于海伦公式计算不规则四边形的面积
def helen_formula(coord):
    coord = np.array(coord).reshape((4, 2))
    # 计算各边的欧式距离
    dis_1 = cal_distance(coord[0], coord[1])
    dis_2 = cal_distance(coord[1], coord[2])
    dis_3 = cal_distance(coord[2], coord[3])
    dis_5 = cal_distance(coord[3], coord[1])
    dis_4 = cal_distance(coord[0], coord[3])
    p1 = (dis_1 + dis_4 + dis_5) * 0.5
    p2 = (dis_2 + dis_3 + dis_5) * 0.5
    # 计算两个三角形的面积
    area1 = np.sqrt(p1 * (p1 - dis_1) * (p1 - dis_4) * (p1 - dis_5))
    area2 = np.sqrt(p2 * (p2 - dis_2) * (p2 - dis_3) * (p2 - dis_5))
    return area1 + area2


# 基于向量积计算不规则四边形的面积
def vector_product(coord):
    coord = np.array(coord).reshape((4, 2))
    temp_det = 0
    for idx in range(3):
        temp = np.array([coord[idx], coord[idx + 1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1], coord[0]]))
    return temp_det * 0.5


coord = [2, 2, 4, 2, 4, 4, 2, 4]
helen_result = helen_formula(coord)
vector_result = vector_product(coord)
print("the result of helen formula:", helen_result)
print("the result of vector product:", vector_result)

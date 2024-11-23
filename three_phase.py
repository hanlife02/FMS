# False fitting of the three-phase region of the ternary phase diagram

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'Microsoft YaHei', 'weight': 'bold'}
plt.rc('font', **font)

# 生成等边三角形顶点
def equilateral_triangle(side_length):
    height = np.sqrt(3) / 2 * side_length 
    return np.array([
        [0, 0, 0], 
        [side_length, 0, 0],
        [side_length / 2, height, 0]  
    ])

# 生成三棱柱顶点
def triangular_prism(side_length, height):
    base = equilateral_triangle(side_length)  # 底面三角形
    top = base + np.array([0, 0, height])  # 顶面三角形通过在 z 方向上平移生成
    return np.vstack([base, top])  # 合并底面和顶面顶点

# 生成三维贝塞尔曲线
def bezier_curve_3d(p0, p1, p2, num=100):
    t = np.linspace(0, 1, num)  # 参数 t 在 [0, 1] 之间均匀分布
    curve_x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    curve_y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    curve_z = (1 - t)**2 * p0[2] + 2 * (1 - t) * t * p1[2] + t**2 * p2[2]
    return curve_x, curve_y, curve_z

# 在曲线上找到最接近目标高度的点的函数
def find_point_at_height(curve_x, curve_y, curve_z, target_height):
    idx = np.abs(curve_z - target_height).argmin()  # 找到 z 值最接近目标高度的点索引
    return curve_x[idx], curve_y[idx], curve_z[idx]

# 三棱柱和贝塞尔曲线的参数
side_length = 5
height = 40  # 三棱柱高度
vertices = triangular_prism(side_length, height)

# 三条贝塞尔曲线的控制点
p0 = (1.5, 0, 38)  # 起点
p2 = (1, 1 * np.sqrt(3), 4)  # 终点
p1 = (0.8, 0.3, 15)  # 控制点

p3 = (3.5, 0, 38)  
p5 = (1.8, 1.8 * np.sqrt(3), 4)  
p4 = (3.4, 0.3, 10) 

p6 = (2.7, 0, 38)  
p8 = (1.4, 1.4 * np.sqrt(3), 4)  
p7 = (2, 0.3, 35) 

curve_x_1, curve_y_1, curve_z_1 = bezier_curve_3d(p0, p1, p2)
curve_x_2, curve_y_2, curve_z_2 = bezier_curve_3d(p3, p4, p5)
curve_x_3, curve_y_3, curve_z_3 = bezier_curve_3d(p6, p7, p8)

# 三角形高度
target_heights = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]  

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三棱柱
faces = [
    [vertices[0], vertices[1], vertices[2]],  # 底面
    [vertices[3], vertices[4], vertices[5]],  # 顶面
    [vertices[0], vertices[1], vertices[4], vertices[3]],  # 侧面1
    [vertices[1], vertices[2], vertices[5], vertices[4]],  # 侧面2
    [vertices[2], vertices[0], vertices[3], vertices[5]]   # 侧面3
]
poly3d = Poly3DCollection(faces, alpha=0.25, linewidths=1, edgecolors='r')
ax.add_collection3d(poly3d)

# 绘制三条曲线
ax.plot(curve_x_1, curve_y_1, curve_z_1, label="拟合线1", color="blue", linewidth=2)
ax.plot(curve_x_2, curve_y_2, curve_z_2, label="拟合线2", color="blue", linewidth=2)
ax.plot(curve_x_3, curve_y_3, curve_z_3, label="拟合线3", color="blue", linewidth=2)

ax.scatter(*zip(p0, p2, p3, p5, p6, p8), color="red", label="端点")

# 遍历多个高度，绘制三角形
for h in target_heights:
    # 在每条曲线上找到目标高度的点
    point1 = find_point_at_height(curve_x_1, curve_y_1, curve_z_1, h)
    point2 = find_point_at_height(curve_x_2, curve_y_2, curve_z_2, h)
    point3 = find_point_at_height(curve_x_3, curve_y_3, curve_z_3, h)

    # 三角形的顶点
    triangle = [point1, point2, point3]

    # 添加三角形到图中
    tri_poly = Poly3DCollection([triangle], color='green', alpha=0.5)
    ax.add_collection3d(tri_poly)

    # 绘制三角形顶点
    ax.scatter(*zip(*triangle), color="green", s=50)

# 设置坐标轴范围
ax.set_xlim([0, side_length])
ax.set_ylim([0, side_length])
ax.set_zlim([0, height])  # 设置 z 轴范围以适应三棱柱高度


# 去掉坐标轴，添加标题
ax.set_axis_off()
ax.legend()
plt.title("假拟合三相区")
plt.show()

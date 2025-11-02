import scipy.io as sio
import numpy as np

# 加载SITE1.mat文件
mat_data = sio.loadmat('SITE1-17/SITE1.mat')  # 确保路径正确（文件夹+文件名是否匹配）

# 1. 打印所有变量名，筛选实际数据变量（排除系统变量）
print("所有变量名：", mat_data.keys())
# 系统变量通常是'__header__', '__version__', '__globals__'，剩下的是数据变量（如'features', 'conn', 'labels'等）

# 2. 替换为你从上面输出中看到的有意义的变量名（比如'X'、'data'、'connectivity'等）
target_var = '替换成实际变量名'  # 例如：如果看到'connectivity'，就改成target_var = 'connectivity'

if target_var in mat_data.keys():
    # 3. 查看数据形状（判断节点类型的核心）
    data = mat_data[target_var]
    print(f"变量'{target_var}'的形状：", data.shape)  # 输出类似(50, 200)或(90, 100)

    # 4. 查看前2个样本的特征（辅助判断）
    print("前2行数据示例：\n", data[:2, :] if data.ndim >= 2 else data[:2])

# 5. 检查是否有标签变量（如'labels'、'diagnosis'等，根据实际变量名修改）
label_var = '替换成标签变量名'  # 例如：如果看到'labels'，就改成label_var = 'labels'
if label_var in mat_data.keys():
    labels = mat_data[label_var]
    print("标签形状：", labels.shape)
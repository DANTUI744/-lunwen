import scipy.io as sio

# 1. 加载1个.mat文件（选任意一个，如SITE1.mat）
file_path = "F:\\github\\ACGA-main\\SITE1-17\\SITE1.mat"  # 替换为你的文件路径
mat_data = sio.loadmat(file_path)

# 2. 打印文件中所有的变量名（关键：看是否有疑似脑区标签的变量）
print("该.mat文件包含的所有变量：")
for idx, var_name in enumerate(mat_data.keys(), 1):
    # 排除.mat文件默认的系统变量（如__header__, __version__）
    if not var_name.startswith("__"):
        print(f"  {idx}. 变量名：{var_name}")
        # 打印变量的形状和前3个元素（快速判断内容）
        var_value = mat_data[var_name]
        print(f"     形状：{var_value.shape}，前3个元素：{var_value[:3] if var_value.size > 3 else var_value}")
        print("-" * 50)
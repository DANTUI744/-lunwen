import torch
import pickle
import scipy.sparse as sp  # 用于构造coo稀疏矩阵
import os

# -------------------------- 1. 加载已分割的图数据（内容不变）
split_graph_path = "F:/github/ACGA-main/population_graph_split.pt"
graph = torch.load(split_graph_path)

# 提取核心数据（与之前完全一致）
features = graph.x.numpy()
labels = graph.y.numpy()
edge_index = graph.edge_index.numpy()  # 形状：(2, E)，E是边数
edge_weight = graph.edge_attr.numpy()  # 形状：(E,)

# 提取分割索引（列表格式，适配模型的tvt_nids[0]/1/2）
train_ids = graph.train_mask.nonzero().squeeze().numpy()
val_ids = graph.val_mask.nonzero().squeeze().numpy()
test_ids = graph.test_mask.nonzero().squeeze().numpy()
tvt_nids = [train_ids, val_ids, test_ids]

# -------------------------- 2. 按coo_matrix要求构造邻接矩阵（核心修改）
# coo_matrix的正确格式：(data, (row_ind, col_ind)) + 形状(节点数×节点数)
num_nodes = graph.num_nodes  # 总节点数（2341）
row = edge_index[0]  # 边的起点索引（第一行）
col = edge_index[1]  # 边的终点索引（第二行）
data = edge_weight   # 边的权重

# 直接构造coo格式的稀疏矩阵
adj_coo = sp.coo_matrix(
    (data, (row, col)),  # 权重+行列索引
    shape=(num_nodes, num_nodes)  # 矩阵形状（必须指定）
)

# -------------------------- 3. 保存所有数据（邻接矩阵用coo格式）
save_dir = "F:/github/ACGA-main/data_new/graphs/"
os.makedirs(save_dir, exist_ok=True)

# 保存邻接矩阵（直接存coo稀疏矩阵，模型可直接解析）
with open(f"{save_dir}/population_adj.pkl", "wb") as f:
    pickle.dump(adj_coo, f)  # 不再存元组，而是存构造好的coo矩阵

# 其他数据保存（与之前一致，确保格式正确）
with open(f"{save_dir}/population_features.pkl", "wb") as f:
    pickle.dump(features, f)

with open(f"{save_dir}/population_labels.pkl", "wb") as f:
    pickle.dump(labels, f)

with open(f"{save_dir}/population_tvt_nids.pkl", "wb") as f:
    pickle.dump(tvt_nids, f)

print(f"数据集已按模型要求重新生成，邻接矩阵为coo格式，保存至：{save_dir}")
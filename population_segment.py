import torch
from sklearn.model_selection import train_test_split

# -------------------------- 1. 加载原始population图（必须先存在）
# 注意：这里要加载的是之前生成的原始图（未分割），路径必须正确
raw_graph_path = "full_mdd_population_graph.pt"  # 替换为你的原始图路径
try:
    graph = torch.load(raw_graph_path)
    print(f"成功加载原始图：节点数={graph.num_nodes}，特征维度={graph.x.shape[1]}")
except FileNotFoundError:
    raise FileNotFoundError(f"原始图文件不存在！请检查路径：{raw_graph_path}")

# -------------------------- 2. 执行图分割（添加train/val/test mask）
def split_population_graph(graph, train_ratio=0.7, val_ratio=0.1, seed=42):
    num_nodes = graph.num_nodes
    indices = torch.arange(num_nodes)
    labels = graph.y.numpy()  # 假设标签是0/1的张量

    # 分层抽样分割训练集和临时集（保持标签比例）
    train_idx, temp_idx, _, _ = train_test_split(
        indices, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )

    # 分割验证集和测试集
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, labels[temp_idx],
        train_size=val_ratio/(val_ratio + (1-train_ratio-val_ratio)),
        stratify=labels[temp_idx],
        random_state=seed
    )

    # 创建mask
    graph.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph.train_mask[train_idx] = True
    graph.val_mask[val_idx] = True
    graph.test_mask[test_idx] = True

    # 验证分割结果
    print(f"\n分割结果：")
    print(f"训练集：{graph.train_mask.sum().item()}个节点")
    print(f"验证集：{graph.val_mask.sum().item()}个节点")
    print(f"测试集：{graph.test_mask.sum().item()}个节点")
    return graph

# 执行分割
graph_split = split_population_graph(graph)

# -------------------------- 3. 保存分割后的图（关键步骤）
save_path = "F:/github/ACGA-main/population_graph_split.pt"  # 保存路径
torch.save(graph_split, save_path)
print(f"\n分割后的图已保存至：{save_path}")
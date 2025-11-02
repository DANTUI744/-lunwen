import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from neuroCombat import neuroCombat


def unify_brain_regions(brain_mat, target_regions=200):
    """统一脑区数量（保留有效信息，避免虚假维度）"""
    original_regions = brain_mat.shape[0]  # 原始脑区数（行维度）
    features = brain_mat.shape[1]  # 特征数（列维度，如1833）

    # 情况1：原始脑区数 == 目标值 → 直接返回
    if original_regions == target_regions:
        return brain_mat

    # 情况2：原始脑区数 > 目标值 → 保留方差大的脑区（信息更丰富）
    elif original_regions > target_regions:
        # 计算每个脑区的特征方差（方差越大，信息越丰富）
        roi_var = np.var(brain_mat, axis=1)  # 形状：(原始脑区数,)
        # 按方差降序排序，取前target_regions个脑区
        top_indices = np.argsort(roi_var)[-target_regions:]  # 最大方差的脑区索引
        top_indices.sort()  # 保持脑区顺序
        return brain_mat[top_indices]  # 保留的脑区矩阵

    # 情况3：原始脑区数 < 目标值 → 用同类脑区均值补充
    else:
        supplement_count = target_regions - original_regions  # 需要补充的脑区数
        # 按功能相似性分组（简化为按索引分段，可根据实际调整）
        n_groups = 5  # 假设分为5个功能网络（如默认网络、视觉网络等）
        group_size = original_regions // n_groups
        groups = [brain_mat[i * group_size: (i + 1) * group_size] for i in range(n_groups)
                  if (i + 1) * group_size <= original_regions]
        # 用每组均值循环补充（确保补充的脑区与原始功能一致）
        supplements = []
        for i in range(supplement_count):
            group_idx = i % len(groups)  # 循环使用各组均值
            supplements.append(np.mean(groups[group_idx], axis=0))  # 组内均值作为补充脑区
        # 拼接原始脑区和补充脑区 → 目标维度
        return np.vstack([brain_mat] + supplements)


def load_all_subjects(data_dir):
    """加载所有文件的被试，统一脑区并收集数据"""
    mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]
    if not mat_files:
        raise ValueError(f"未在{data_dir}找到.mat文件")

    all_features = []  # 存储所有被试的脑区数据（统一后）
    all_labels = []  # 存储所有被试的标签（如0/1）
    all_sites = []  # 存储被试来源文件（用于ComBat）

    for file_idx, file in enumerate(mat_files, 1):
        file_path = os.path.join(data_dir, file)
        print(f"处理文件 [{file_idx}/{len(mat_files)}]：{file}")

        # 加载文件
        mat_data = loadmat(file_path)
        # 你的数据中脑区变量是"A"，标签是"lab"（修正变量名）
        if "A" not in mat_data or "lab" not in mat_data:
            print(f"  ❌ 缺少'A'或'lab'变量，跳过")
            continue

        # 提取被试数据和标签（"A"是(1, N)的列表，每个元素是(脑区数, 特征数)）
        subjects = mat_data["A"][0]  # 修正为"A"
        labels = mat_data["lab"].ravel()  # 被试标签（形状：(N,)）

        # 验证被试数与标签数一致
        if len(subjects) != len(labels):
            print(f"  ⚠️ 被试数({len(subjects)})与标签数({len(labels)})不匹配，截断标签")
            labels = labels[:len(subjects)]

        # 逐个处理被试，统一脑区并收集（修正缩进：放在循环内，与文件处理对齐）
        file_valid_count = 0  # 记录当前文件的有效被试数
        for subj_idx, (brain_mat, label) in enumerate(zip(subjects, labels)):
            # 跳过特征数异常的被试（假设正常为1833）
            if brain_mat.shape[1] != 1833:
                print(f"  ⚠️ 被试{subj_idx}特征数异常（{brain_mat.shape[1]}≠1833），跳过")
                continue

            # 统一脑区数量
            try:
                unified_brain = unify_brain_regions(brain_mat, target_regions=200)
            except Exception as e:
                print(f"  ⚠️ 被试{subj_idx}脑区统一失败：{str(e)}，跳过")
                continue

            # 聚合为被试级特征（按脑区取均值）
            subj_feature = np.mean(unified_brain, axis=0)  # 形状：(1833,)

            # 验证标签合法性（0/1）
            if label not in [0, 1]:
                print(f"  ⚠️ 被试{subj_idx}标签异常（{label}≠0/1），跳过")
                continue

            # 收集有效数据
            all_features.append(subj_feature)
            all_labels.append(label)
            all_sites.append(file)
            file_valid_count += 1

        print(f"  ✅ 完成，该文件有效被试数：{file_valid_count}/{len(subjects)}")  # 修正计数逻辑

    # 转换为数组格式（修正缩进：放在所有文件处理完之后）
    if not all_features:
        raise ValueError("未加载到任何有效被试数据")
    all_features = np.array(all_features)  # 形状：(总被试数, 1833)
    all_labels = np.array(all_labels)  # 形状：(总被试数,)
    print(f"\n全量数据加载完成：总有效被试数={all_features.shape[0]}")
    return all_features, all_labels, all_sites


# 修正：将process_features移出load_all_subjects，放在全局作用域
def process_features(features, sites):
    """根治奇异矩阵：仅传原始站点ID，让库自动处理编码（控制drop_first）"""
    # 1. 站点ID编码（仅保留原始ID，不手动生成独热编码）
    unique_sites = list(set(sites))
    n_sites = len(unique_sites)
    site_ids = np.array([unique_sites.index(s) for s in sites])  # 0~23（原始站点ID）

    # 检查站点样本量（确保≥2，避免效应无法估计）
    site_counts = pd.Series(site_ids).value_counts().sort_index()
    print("各站点被试数量：")
    print(site_counts)
    if (site_counts < 2).any():
        print("⚠️ 警告：存在被试数≤1的站点，可能导致校正失败")

    # 2. 协变量仅含原始站点ID（不手动加独热编码，避免冗余）
    covars = pd.DataFrame({"batch": site_ids})  # 形状：(2341, 1)（仅1列）
    print(f"协变量形状：{covars.shape}（应是(2341, 1)）")

    # 3. 准备数据（特征数×被试数）
    data_for_combat = features.T  # 形状：(1833, 2341)

    # 4. 调用ComBat：让库自动对batch列编码，强制drop_first=True消除虚拟变量陷阱
    print("\n开始ComBat站点效应校正（根治奇异矩阵）...")
    # 关键：通过修改库的编码逻辑，对categorical列做独热编码时drop_first
    # （最新版neuroCombat可通过`drop_first=True`参数控制，若不支持则手动处理）
    combat_result = neuroCombat(
        dat=data_for_combat,
        covars=covars,
        batch_col="batch",  # 仅指定原始站点ID列为batch
        categorical_cols=["batch"],  # 明确batch是分类变量
        drop_first=True  # 核心参数：对categorical列编码时删除第一列（24→23列）
    )

    # 5. 转回被试×特征格式并标准化
    combat_data = combat_result["data"].T  # 形状：(2341, 1833)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combat_data)
    print(
        f"特征处理完成：形状={scaled_features.shape}，均值≈{scaled_features.mean():.2f}，标准差≈{scaled_features.std():.2f}")
    return scaled_features
def build_population_graph(features, labels, top_k=20):
    """构建population图（节点=被试，边=被试间相似度）"""
    n_subjects = features.shape[0]
    print(f"\n开始构建population图：{n_subjects}个节点（被试）")

    # 计算被试间余弦相似度
    sim_matrix = cosine_similarity(features)  # 形状：(n_subjects, n_subjects)

    # 用top-k近邻生成边
    edge_index = []
    edge_weight = []
    for i in range(n_subjects):
        sim = sim_matrix[i].copy()
        sim[i] = -1  # 排除自身
        top_k_indices = np.argsort(sim)[-top_k:]  # 最相似的top_k个被试
        for j in top_k_indices:
            edge_index.append([i, j])
            edge_weight.append(sim[j])

    # 转换为PyTorch Geometric格式
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    print(f"图构建完成：节点数={graph.num_nodes}，边数={graph.num_edges}（每个节点保留{top_k}条边）")
    return graph


# 修正：将main移出load_all_subjects
def main(data_dir, save_path="population_graph.pt"):
    # 步骤1：加载并整合所有被试
    features, labels, sites = load_all_subjects(data_dir)

    # 步骤2：处理特征（校正+标准化）
    processed_features = process_features(features, sites)

    # 步骤3：构建population图
    graph = build_population_graph(processed_features, labels, top_k=20)

    # 保存图
    torch.save(graph, save_path)
    print(f"\n图已保存至：{save_path}")


# 运行主函数
if __name__ == "__main__":
    DATA_DIR = "F:\\github\\ACGA-main\\SITE1-17"  # 你的数据文件夹路径
    main(DATA_DIR)
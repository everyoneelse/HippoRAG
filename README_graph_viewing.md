# HippoRAG 知识图谱查看指南

本指南介绍了如何查看和分析 HippoRAG 系统构建的知识图谱。

## 📋 目录

1. [快速开始](#快速开始)
2. [图谱查看方法](#图谱查看方法)
3. [可视化功能](#可视化功能)
4. [高级功能](#高级功能)
5. [故障排除](#故障排除)

## 🚀 快速开始

### 方法1: 使用简单查看器

```bash
# 查看已构建的图谱基本信息
python simple_graph_viewer.py <save_dir> <llm_name> <embedding_name>

# 示例
python simple_graph_viewer.py outputs gpt-4o-mini nvidia/NV-Embed-v2
```

### 方法2: 运行完整示例

```bash
# 构建示例图谱并查看（需要OpenAI API Key）
python example_view_graph.py
```

## 📊 图谱查看方法

### 1. 基本统计信息

HippoRAG 图谱包含以下统计信息：

- **实体节点数量**: 从文档中提取的实体（人名、地名、概念等）
- **文档节点数量**: 原始文档块的数量
- **总节点数量**: 实体节点 + 文档节点
- **提取的三元组**: 从文档中提取的知识三元组 [主语, 谓语, 宾语]
- **文档关系**: 连接文档节点的关系
- **同义词关系**: 相似实体之间的连接
- **总关系数量**: 图谱中所有边的总数

### 2. 在代码中查看图谱

```python
from src.hipporag.HippoRAG import HippoRAG

# 初始化已存在的HippoRAG实例
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# 获取图谱统计信息
graph_info = hipporag.get_graph_info()
print(f"节点数量: {graph_info['num_total_nodes']}")
print(f"关系数量: {graph_info['num_total_triples']}")

# 查看实体节点
entity_nodes = hipporag.entity_embedding_store.get_all_ids()
print("实体节点:", entity_nodes[:10])

# 查看文档节点
passage_nodes = hipporag.chunk_embedding_store.get_all_ids()
print("文档节点:", passage_nodes[:5])

# 查看提取的三元组
facts = hipporag.fact_embedding_store.get_all_ids()
print("三元组:", facts[:10])
```

### 3. 使用高级查看工具

```bash
# 查看详细统计和导出数据
python visualize_graph.py --save_dir outputs --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2 --export

# 搜索特定关键词的节点
python visualize_graph.py --save_dir outputs --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2 --search "politician"

# 可视化以特定节点为中心的子图
python visualize_graph.py --save_dir outputs --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2 --visualize "Obama,president"
```

## 🎨 可视化功能

### 安装可视化依赖

```bash
pip install matplotlib networkx
```

### 生成图谱可视化

```python
from visualize_graph import GraphViewer

# 创建查看器实例
viewer = GraphViewer(hipporag)

# 可视化子图（以指定节点为中心）
viewer.visualize_subgraph(
    center_nodes=['Obama', 'president'], 
    max_nodes=50, 
    save_path='my_graph.png'
)
```

## 🔧 高级功能

### 1. 导出图谱数据

```bash
# 导出所有图谱数据到JSON文件
python visualize_graph.py --save_dir outputs --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2 --export
```

导出的文件包括：
- `nodes.json`: 所有节点信息
- `edges.json`: 所有边信息  
- `statistics.json`: 图谱统计数据

### 2. 搜索和过滤

```python
# 搜索包含特定关键词的节点
viewer = GraphViewer(hipporag)
found_nodes = viewer.search_nodes("politician", limit=20)
```

### 3. 图谱文件位置

HippoRAG 将图谱保存在以下位置：

```
outputs/
└── <dataset_name>/
    └── <llm_name>_<embedding_name>/
        ├── graph.pickle          # igraph图谱文件
        ├── entity_store.pkl      # 实体嵌入存储
        ├── chunk_store.pkl       # 文档嵌入存储
        └── fact_store.pkl        # 三元组嵌入存储
```

### 4. 直接访问图谱对象

```python
# 访问底层igraph对象
graph = hipporag.graph

# 查看图谱基本信息
print(f"节点数: {graph.vcount()}")
print(f"边数: {graph.ecount()}")

# 获取所有节点名称
node_names = graph.vs["name"] if "name" in graph.vs.attributes() else []
print("节点名称:", node_names[:10])
```

## ❗ 故障排除

### 1. 图谱文件不存在

**错误**: `FileNotFoundError: graph.pickle not found`

**解决方案**: 
- 确保已经运行过 `hipporag.index(docs)` 构建图谱
- 检查 `save_dir` 路径是否正确
- 确认 LLM 和嵌入模型名称匹配

### 2. 空图谱

**现象**: 所有统计数据为0

**可能原因**:
- OpenIE 提取失败
- 文档内容过于简单
- API 调用失败

**解决方案**:
- 检查日志文件
- 确认 API Key 设置正确
- 尝试使用更复杂的示例文档

### 3. 可视化失败

**错误**: `需要安装 matplotlib 和 networkx`

**解决方案**:
```bash
pip install matplotlib networkx
```

### 4. 内存不足

**现象**: 大型图谱加载失败

**解决方案**:
- 使用 `limit` 参数限制显示数量
- 分批处理大型图谱
- 增加系统内存

## 📝 示例输出

```
🚀 HippoRAG 简单图谱查看器
📁 保存目录: outputs
🤖 LLM模型: gpt-4o-mini
📊 嵌入模型: nvidia/NV-Embed-v2
============================================================

📊 知识图谱统计信息:
  🔸 实体节点: 45 个
  🔸 文档节点: 14 个
  🔸 总节点数: 59 个
  🔸 提取三元组: 38 个
  🔸 文档关系: 14 个
  🔸 同义关系: 12 个
  🔸 总关系数: 64 个

🔸 实体节点示例:
  1. Oliver Badman
  2. politician
  3. UK
  4. George Rankin
  5. Scotland
  ... 还有 40 个实体节点

🔸 文档节点示例:
  1. Oliver Badman is a politician from the UK.
  2. George Rankin is a politician from Scotland.
  3. Thomas Marwick is a politician from Ireland.
  ... 还有 11 个文档节点
```

## 🔍 更多信息

- 查看 `main.py` 了解如何在实验中使用图谱
- 参考 `src/hipporag/HippoRAG.py` 了解图谱构建细节
- 阅读论文了解 HippoRAG 的理论背景

## 📞 支持

如果遇到问题，请：
1. 检查日志输出
2. 确认环境配置正确
3. 参考故障排除部分
4. 提交 GitHub Issue
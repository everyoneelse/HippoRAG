# HippoRAG 知识图谱保存功能使用指南

## 概述

HippoRAG 项目已经实现了完善的知识图谱保存和导出功能。本指南将详细介绍如何使用这些功能来保存、导出和重新加载知识图谱。

## 🎯 主要功能

### 1. 自动保存功能（已有）
HippoRAG 在构建知识图谱时会自动保存以下内容：
- **图结构**: 保存为 igraph pickle 格式 (`graph.pickle`)
- **OpenIE 结果**: 保存为 JSON 格式 (`openie_results_ner_{llm_name}.json`)
- **嵌入向量**: 分别保存实体、事实和文档的嵌入向量

### 2. 新增导出功能

#### 2.1 导出知识图谱到不同格式
```python
# 导出为 JSON 格式（包含详细节点和边信息）
json_path = hipporag.export_knowledge_graph('json', 'my_graph.json')

# 导出为 GraphML 格式（可用于 Gephi 等图分析工具）
graphml_path = hipporag.export_knowledge_graph('graphml', 'my_graph.graphml')

# 导出为 GML 格式
gml_path = hipporag.export_knowledge_graph('gml', 'my_graph.gml')

# 导出为边列表格式
edgelist_path = hipporag.export_knowledge_graph('edgelist', 'my_graph.txt')
```

#### 2.2 导出 OpenIE 结果
```python
# 导出 OpenIE 提取的实体和三元组
openie_path = hipporag.export_openie_results('openie_export.json')
```

#### 2.3 导出完整知识库
```python
# 一键导出所有相关文件
saved_files = hipporag.save_complete_knowledge_base('export_directory')
```

## 📁 文件结构说明

### 工作目录结构
```
{save_dir}/{llm_model}_{embedding_model}/
├── graph.pickle                    # 图结构（igraph格式）
├── chunk_embeddings/               # 文档嵌入向量
├── entity_embeddings/              # 实体嵌入向量
├── fact_embeddings/                # 事实嵌入向量
└── openie_results_ner_{llm}.json   # OpenIE结果
```

### 导出文件结构
```
export_directory/
├── knowledge_base_export_{timestamp}/
│   ├── graph.pickle                # 原始图结构
│   ├── graph.json                  # JSON格式图谱
│   ├── graph.graphml               # GraphML格式图谱
│   ├── openie_results.json         # OpenIE结果
│   ├── embeddings_info.json        # 嵌入向量信息
│   └── export_summary.json         # 导出摘要
```

## 🚀 使用示例

### 基本使用
```python
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# 配置
config = BaseConfig(
    save_dir='outputs',
    llm_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2',  # 或使用本地路径
    dataset='my_dataset'
)

# 初始化
hipporag = HippoRAG(global_config=config)

# 构建知识图谱
docs = ["文档1", "文档2", "文档3"]
hipporag.index(docs)

# 导出知识图谱
hipporag.export_knowledge_graph('json', 'my_knowledge_graph.json')
```

### 使用本地模型
```python
# 如果您的模型下载在本地
config = BaseConfig(
    save_dir='outputs',
    llm_name='gpt-4o-mini',
    embedding_model_name='/path/to/your/nvidia-NV-Embed-v2',  # 本地路径
    dataset='my_dataset'
)
```

### 完整导出示例
```python
# 导出完整知识库
saved_files = hipporag.save_complete_knowledge_base()

print("导出的文件:")
for file_type, file_path in saved_files.items():
    print(f"- {file_type}: {file_path}")
```

### 重新加载知识图谱
```python
# 使用相同配置重新加载
config_reload = BaseConfig(
    save_dir='outputs',
    llm_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2',
    dataset='my_dataset',
    force_index_from_scratch=False  # 使用已保存的数据
)

hipporag_reloaded = HippoRAG(global_config=config_reload)

# 测试检索
queries = ["查询问题"]
results = hipporag_reloaded.retrieve(queries)
```

## 📊 导出格式详解

### 1. JSON 格式
包含完整的图结构信息：
```json
{
  "metadata": {
    "created_at": "2024-01-01T00:00:00",
    "llm_model": "gpt-4o-mini",
    "embedding_model": "nvidia/NV-Embed-v2",
    "statistics": {...}
  },
  "nodes": [
    {
      "id": 0,
      "name": "entity-hash",
      "type": "entity",
      "content": "实体内容",
      "attributes": {...}
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "weight": 1.0,
      "attributes": {...}
    }
  ]
}
```

### 2. GraphML 格式
标准的图交换格式，可以被以下工具读取：
- Gephi（图可视化）
- Cytoscape（网络分析）
- NetworkX（Python图库）
- igraph（R/Python）

### 3. OpenIE 结果格式
```json
{
  "metadata": {
    "created_at": "2024-01-01T00:00:00",
    "total_documents": 100,
    "total_entities": 500,
    "total_triples": 800
  },
  "documents": [
    {
      "idx": "chunk-hash",
      "passage": "原文内容",
      "extracted_entities": ["实体1", "实体2"],
      "extracted_triples": [["主语", "谓语", "宾语"]]
    }
  ]
}
```

## 🛠️ 高级功能

### 自定义导出路径
```python
# 指定导出路径
custom_path = "/path/to/my/export/graph.json"
hipporag.export_knowledge_graph('json', custom_path)
```

### 批量导出多种格式
```python
formats = ['json', 'graphml', 'gml']
for fmt in formats:
    output_path = f"my_graph.{fmt}"
    hipporag.export_knowledge_graph(fmt, output_path)
    print(f"导出 {fmt} 格式完成: {output_path}")
```

## 📈 性能和存储

### 存储空间估算
- **图结构**: 通常几MB到几十MB
- **嵌入向量**: 取决于文档数量和嵌入维度
- **OpenIE结果**: 取决于提取的实体和三元组数量

### 性能优化建议
1. 定期清理不需要的导出文件
2. 使用压缩格式存储大型图谱
3. 根据需要选择合适的导出格式

## 🔧 故障排除

### 常见问题
1. **导出失败**: 检查磁盘空间和文件权限
2. **格式不支持**: 确认使用支持的格式（json, graphml, gml, edgelist, pajek）
3. **文件过大**: 考虑分批导出或使用压缩

### 调试方法
```python
import logging
logging.basicConfig(level=logging.INFO)

# 查看详细日志
hipporag.export_knowledge_graph('json', 'debug_graph.json')
```

## 📝 示例脚本

项目中包含以下示例脚本：
- `demo_save_kg.py`: 基本保存功能演示
- `export_knowledge_graph.py`: 完整导出功能演示

### 运行示例

#### 使用默认模型（从Hugging Face下载）
```bash
python demo_save_kg.py
python export_knowledge_graph.py --dataset sample --export_format json
```

#### 使用本地模型路径
```bash
# 基本演示
python demo_save_kg.py --embedding_path /path/to/your/nvidia-NV-Embed-v2

# 完整导出
python export_knowledge_graph.py \
    --dataset sample \
    --embedding_path /path/to/your/nvidia-NV-Embed-v2 \
    --export_format all
```

### 命令行参数说明
- `--embedding_path`: 指定本地嵌入模型路径
- `--dataset`: 数据集名称
- `--export_format`: 导出格式（json/graphml/gml/all）
- `--export_dir`: 导出目录

## 🎉 总结

HippoRAG 的知识图谱保存功能提供了：
- ✅ 自动保存和加载
- ✅ 多种导出格式支持
- ✅ 完整的元数据记录
- ✅ 灵活的配置选项
- ✅ 详细的使用文档

通过这些功能，您可以轻松地保存、分享和重用构建的知识图谱，为您的RAG应用提供持久化的知识存储能力。
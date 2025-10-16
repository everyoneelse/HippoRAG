# 使用本地模型路径的说明

## 概述

如果您已经将 `nvidia/NV-Embed-v2` 或其他嵌入模型下载到本地，可以通过以下方式在HippoRAG中使用本地模型路径。

## 🔧 修改方法

### 1. 使用命令行参数（推荐）

#### export_knowledge_graph.py
```bash
# 使用本地模型路径
python export_knowledge_graph.py --embedding_path /path/to/your/nvidia-NV-Embed-v2

# 完整示例
python export_knowledge_graph.py \
    --dataset sample \
    --embedding_path /home/user/models/nvidia-NV-Embed-v2 \
    --export_format json
```

#### demo_save_kg.py
```bash
# 使用本地模型路径
python demo_save_kg.py --embedding_path /path/to/your/nvidia-NV-Embed-v2
```

### 2. 直接在代码中修改

如果您想直接修改代码，可以在配置中指定本地路径：

```python
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

# 配置本地模型路径
config = BaseConfig(
    save_dir='outputs',
    llm_name='gpt-4o-mini',
    embedding_model_name='/path/to/your/nvidia-NV-Embed-v2',  # 使用本地路径
    dataset='my_dataset'
)

hipporag = HippoRAG(global_config=config)
```

## 📁 常见的本地模型路径示例

### Linux/Mac
```bash
# 如果模型在用户目录下
--embedding_path /home/username/models/nvidia-NV-Embed-v2

# 如果模型在项目目录下
--embedding_path ./models/nvidia-NV-Embed-v2

# 如果模型在共享目录下
--embedding_path /opt/models/nvidia-NV-Embed-v2
```

### Windows
```bash
# 如果模型在用户目录下
--embedding_path C:\Users\username\models\nvidia-NV-Embed-v2

# 如果模型在项目目录下
--embedding_path .\models\nvidia-NV-Embed-v2
```

## 🔍 如何找到您的模型路径

### 1. 通过Hugging Face缓存
如果您之前使用过该模型，它可能被缓存在：

```bash
# Linux/Mac
~/.cache/huggingface/hub/models--nvidia--NV-Embed-v2

# Windows
C:\Users\{username}\.cache\huggingface\hub\models--nvidia--NV-Embed-v2
```

### 2. 手动下载的模型
如果您手动下载了模型，路径就是您保存模型文件的目录。

### 3. 检查模型文件
确保您的模型目录包含以下文件：
```
nvidia-NV-Embed-v2/
├── config.json
├── pytorch_model.bin 或 model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

## 🚀 完整使用示例

### 示例1：导出知识图谱（使用本地模型）
```bash
python export_knowledge_graph.py \
    --dataset sample \
    --llm_name gpt-4o-mini \
    --embedding_path /home/user/models/nvidia-NV-Embed-v2 \
    --export_format all \
    --export_dir my_exports
```

### 示例2：演示保存功能（使用本地模型）
```bash
python demo_save_kg.py --embedding_path /home/user/models/nvidia-NV-Embed-v2
```

### 示例3：在main.py中使用本地模型
```bash
python main.py \
    --dataset sample \
    --llm_name gpt-4o-mini \
    --embedding_name /home/user/models/nvidia-NV-Embed-v2
```

## ⚠️ 注意事项

1. **路径格式**：
   - 使用绝对路径更可靠
   - 确保路径中没有空格，或者用引号包围路径
   - Windows用户注意使用正确的路径分隔符

2. **权限问题**：
   - 确保Python进程有读取模型文件的权限
   - 如果模型在系统目录下，可能需要管理员权限

3. **模型完整性**：
   - 确保所有必需的模型文件都存在
   - 检查文件是否完整下载

4. **兼容性**：
   - 确保本地模型版本与HippoRAG兼容
   - 建议使用官方发布的模型版本

## 🔧 故障排除

### 常见错误及解决方法

1. **找不到模型文件**
   ```
   错误: No such file or directory
   解决: 检查路径是否正确，使用绝对路径
   ```

2. **权限被拒绝**
   ```
   错误: Permission denied
   解决: 检查文件权限，或使用sudo运行（不推荐）
   ```

3. **模型加载失败**
   ```
   错误: Can't load model
   解决: 检查模型文件完整性，重新下载模型
   ```

### 验证模型路径
您可以使用以下Python代码验证模型路径是否正确：

```python
import os
from transformers import AutoModel, AutoTokenizer

model_path = "/path/to/your/nvidia-NV-Embed-v2"

# 检查路径是否存在
if os.path.exists(model_path):
    print(f"✅ 模型路径存在: {model_path}")
    
    # 尝试加载模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
else:
    print(f"❌ 模型路径不存在: {model_path}")
```

## 📝 环境变量方式（可选）

您也可以通过设置环境变量来指定模型路径：

```bash
# 设置环境变量
export EMBEDDING_MODEL_PATH="/path/to/your/nvidia-NV-Embed-v2"

# 然后在代码中使用
python export_knowledge_graph.py --embedding_path $EMBEDDING_MODEL_PATH
```

这样您就可以灵活地使用本地下载的嵌入模型了！🎉
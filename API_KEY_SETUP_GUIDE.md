# HippoRAG API Key 设置指南

## 概述

HippoRAG 支持多种 LLM 服务，需要相应的 API key 来访问这些服务。本指南将详细说明如何设置各种 API key。

## 🔑 支持的 API 服务

### 1. OpenAI API
- **用途**: GPT-4, GPT-4o-mini 等模型
- **环境变量**: `OPENAI_API_KEY`

### 2. Azure OpenAI
- **用途**: Azure 部署的 OpenAI 模型
- **环境变量**: `AZURE_OPENAI_API_KEY`

### 3. 本地部署的模型
- **用途**: vLLM, Ollama 等本地服务
- **通常不需要**: API key（或使用占位符）

## 🚀 设置方法

### 方法1: 环境变量设置（推荐）

#### Linux/Mac
```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# 设置 Azure OpenAI API Key（如果使用 Azure）
export AZURE_OPENAI_API_KEY="your-azure-api-key-here"

# 验证设置
echo $OPENAI_API_KEY
```

#### Windows (PowerShell)
```powershell
# 设置 OpenAI API Key
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"

# 设置 Azure OpenAI API Key（如果使用 Azure）
$env:AZURE_OPENAI_API_KEY="your-azure-api-key-here"

# 验证设置
echo $env:OPENAI_API_KEY
```

#### Windows (命令提示符)
```cmd
# 设置 OpenAI API Key
set OPENAI_API_KEY=sk-your-openai-api-key-here

# 设置 Azure OpenAI API Key（如果使用 Azure）
set AZURE_OPENAI_API_KEY=your-azure-api-key-here

# 验证设置
echo %OPENAI_API_KEY%
```

### 方法2: .env 文件设置

创建 `.env` 文件在项目根目录：
```bash
# .env 文件内容
OPENAI_API_KEY=sk-your-openai-api-key-here
AZURE_OPENAI_API_KEY=your-azure-api-key-here
HF_TOKEN=your-huggingface-token-here
```

然后在 Python 代码中加载：
```python
from dotenv import load_dotenv
load_dotenv()

# 现在可以使用环境变量了
```

### 方法3: 直接在代码中设置（不推荐）

```python
import os

# 在代码中设置（仅用于测试，不要提交到版本控制）
os.environ["OPENAI_API_KEY"] = "sk-your-openai-api-key-here"
```

### 方法4: 系统级永久设置

#### Linux/Mac - 添加到 shell 配置文件
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export OPENAI_API_KEY="sk-your-openai-api-key-here"' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

#### Windows - 系统环境变量
1. 右键"此电脑" → "属性"
2. "高级系统设置" → "环境变量"
3. 在"用户变量"中添加：
   - 变量名: `OPENAI_API_KEY`
   - 变量值: `sk-your-openai-api-key-here`

## 📝 具体使用场景

### 1. 使用 OpenAI GPT 模型

```bash
# 设置 API key
export OPENAI_API_KEY="sk-your-openai-api-key-here"

# 运行脚本
python export_knowledge_graph.py --llm_name gpt-4o-mini
```

### 2. 使用 Azure OpenAI

```bash
# 设置 Azure API key
export AZURE_OPENAI_API_KEY="your-azure-api-key-here"

# 运行脚本（需要指定 Azure endpoint）
python main.py \
    --llm_name gpt-4o-mini \
    --azure_endpoint "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-15-preview"
```

### 3. 使用本地部署的模型

```bash
# 本地模型通常不需要真实的 API key，但可能需要占位符
export OPENAI_API_KEY="sk-placeholder"

# 运行脚本
python export_knowledge_graph.py \
    --llm_name meta-llama/Llama-3.3-70B-Instruct \
    --llm_base_url http://localhost:8000/v1
```

## 🔧 修改脚本以支持 API key 设置

我来为您的脚本添加 API key 设置功能：

### 修改 export_knowledge_graph.py

```python
def main():
    parser = argparse.ArgumentParser(description="导出HippoRAG知识图谱")
    # ... 其他参数 ...
    parser.add_argument('--openai_api_key', type=str, default=None, 
                       help='OpenAI API Key（也可通过环境变量 OPENAI_API_KEY 设置）')
    parser.add_argument('--azure_api_key', type=str, default=None,
                       help='Azure OpenAI API Key（也可通过环境变量 AZURE_OPENAI_API_KEY 设置）')
    
    args = parser.parse_args()
    
    # 设置 API keys
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
        print("🔑 使用命令行提供的 OpenAI API Key")
    elif not os.getenv('OPENAI_API_KEY'):
        print("⚠️  未设置 OPENAI_API_KEY 环境变量")
    
    if args.azure_api_key:
        os.environ['AZURE_OPENAI_API_KEY'] = args.azure_api_key
        print("🔑 使用命令行提供的 Azure API Key")
```

### 修改 demo_save_kg.py

```python
def demo_save_knowledge_graph(embedding_model_path=None, openai_api_key=None):
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        print("🔑 使用提供的 OpenAI API Key")
    elif not os.getenv('OPENAI_API_KEY'):
        print("⚠️  未设置 OPENAI_API_KEY，如果使用 OpenAI 模型可能会失败")
```

## 🛡️ 安全最佳实践

### 1. 不要硬编码 API key
```python
# ❌ 错误做法
api_key = "sk-your-actual-api-key-here"

# ✅ 正确做法
api_key = os.getenv('OPENAI_API_KEY')
```

### 2. 使用 .gitignore 忽略敏感文件
```gitignore
# .gitignore 文件内容
.env
*.key
api_keys.txt
```

### 3. 使用环境变量验证
```python
import os

def check_api_keys():
    """检查必要的 API keys 是否设置"""
    required_keys = ['OPENAI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ 缺少以下环境变量: {', '.join(missing_keys)}")
        return False
    
    print("✅ 所有必要的 API keys 已设置")
    return True
```

## 🔍 故障排除

### 常见问题

1. **API key 无效**
   ```
   错误: Invalid API key
   解决: 检查 API key 是否正确，是否有相应权限
   ```

2. **环境变量未设置**
   ```
   错误: No API key provided
   解决: 确保设置了正确的环境变量
   ```

3. **权限不足**
   ```
   错误: Insufficient quota
   解决: 检查 API 使用额度和权限
   ```

### 调试方法

```python
import os

def debug_api_keys():
    """调试 API key 设置"""
    keys_to_check = ['OPENAI_API_KEY', 'AZURE_OPENAI_API_KEY']
    
    for key in keys_to_check:
        value = os.getenv(key)
        if value:
            # 只显示前几个和后几个字符，保护隐私
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: 未设置")

# 运行调试
debug_api_keys()
```

## 📚 获取 API Key

### OpenAI API Key
1. 访问 [OpenAI Platform](https://platform.openai.com/)
2. 登录账户
3. 进入 "API Keys" 页面
4. 点击 "Create new secret key"
5. 复制生成的 key（格式：sk-...）

### Azure OpenAI API Key
1. 登录 [Azure Portal](https://portal.azure.com/)
2. 找到您的 OpenAI 资源
3. 在 "Keys and Endpoint" 页面获取 API key

### Hugging Face Token（如果需要）
1. 访问 [Hugging Face](https://huggingface.co/)
2. 登录账户
3. 进入 Settings → Access Tokens
4. 创建新的 token

## 🎯 快速开始

1. **获取 API key**（见上方说明）

2. **设置环境变量**：
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **验证设置**：
   ```bash
   echo $OPENAI_API_KEY
   ```

4. **运行脚本**：
   ```bash
   python export_knowledge_graph.py --dataset sample
   ```

现在您就可以正确设置和使用 API key 了！🎉
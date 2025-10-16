#!/usr/bin/env python3
"""
禁用 transformers 网络验证的多种方法
"""

import os
import json
from pathlib import Path

def method1_environment_variables():
    """方法1: 设置环境变量（最简单）"""
    print("🔧 方法1: 设置环境变量")
    print("-" * 30)
    
    env_vars = {
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
        'TOKENIZERS_PARALLELISM': 'false'
    }
    
    print("在运行 Python 之前设置以下环境变量:")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"export {key}={value}")
    
    print("\n或者在 Python 代码开头添加:")
    print("import os")
    for key, value in env_vars.items():
        print(f"os.environ['{key}'] = '{value}'")

def method2_modify_transformers_config():
    """方法2: 修改 transformers 配置"""
    print("\n🔧 方法2: 修改 transformers 配置")
    print("-" * 30)
    
    config_code = '''
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 禁用网络检查
transformers.utils.hub.HF_HUB_OFFLINE = True

# 或者在加载模型时强制使用本地文件
model = AutoModel.from_pretrained(
    "/path/to/your/model",
    local_files_only=True,
    use_auth_token=False,
    trust_remote_code=True
)
'''
    print("在代码中添加:")
    print(config_code)

def method3_patch_huggingface_hub():
    """方法3: 猴子补丁 huggingface_hub"""
    print("\n🔧 方法3: 猴子补丁 huggingface_hub")
    print("-" * 30)
    
    patch_code = '''
# 在导入 transformers 之前运行
import huggingface_hub
from unittest.mock import patch

# 禁用网络请求
def mock_http_get(*args, **kwargs):
    raise Exception("Network requests disabled")

def mock_http_head(*args, **kwargs):
    raise Exception("Network requests disabled")

# 应用补丁
huggingface_hub.utils._http.http_get = mock_http_get
huggingface_hub.utils._http.http_head = mock_http_head

# 现在可以安全导入 transformers
import transformers
'''
    print("使用猴子补丁:")
    print(patch_code)

def method4_create_offline_wrapper():
    """方法4: 创建离线包装器"""
    print("\n🔧 方法4: 创建离线包装器")
    print("-" * 30)
    
    wrapper_code = '''
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig

class OfflineTransformers:
    """完全离线的 transformers 包装器"""
    
    def __init__(self):
        # 设置离线环境
        os.environ.update({
            'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1',
            'HF_HUB_DISABLE_TELEMETRY': '1'
        })
    
    def load_model(self, model_path, **kwargs):
        """加载本地模型，强制离线模式"""
        kwargs.update({
            'local_files_only': True,
            'use_auth_token': False,
            'trust_remote_code': True,
            'offline': True
        })
        
        try:
            return AutoModel.from_pretrained(model_path, **kwargs)
        except Exception as e:
            print(f"离线加载失败: {e}")
            # 尝试不同的参数组合
            kwargs.pop('offline', None)
            return AutoModel.from_pretrained(model_path, **kwargs)
    
    def load_tokenizer(self, model_path, **kwargs):
        """加载本地分词器"""
        kwargs.update({
            'local_files_only': True,
            'use_auth_token': False
        })
        return AutoTokenizer.from_pretrained(model_path, **kwargs)

# 使用方法
offline_transformers = OfflineTransformers()
model = offline_transformers.load_model("/path/to/your/model")
tokenizer = offline_transformers.load_tokenizer("/path/to/your/model")
'''
    print("创建离线包装器:")
    print(wrapper_code)

def method5_modify_model_config():
    """方法5: 修改模型配置文件"""
    print("\n🔧 方法5: 修改模型配置文件")
    print("-" * 30)
    
    print("在模型目录中创建或修改 config.json，添加:")
    config_example = {
        "auto_map": None,
        "_name_or_path": "local_model",
        "transformers_version": "4.21.0"
    }
    print(json.dumps(config_example, indent=2))
    
    print("\n这样可以避免 transformers 尝试从网络获取配置信息")

def method6_network_isolation():
    """方法6: 网络隔离方法"""
    print("\n🔧 方法6: 网络隔离")
    print("-" * 30)
    
    isolation_code = '''
import socket
from unittest.mock import patch

def disable_network():
    """完全禁用网络连接"""
    def mock_getaddrinfo(*args, **kwargs):
        raise socket.gaierror("Network disabled")
    
    def mock_create_connection(*args, **kwargs):
        raise ConnectionError("Network disabled")
    
    socket.getaddrinfo = mock_getaddrinfo
    socket.create_connection = mock_create_connection

# 在导入 transformers 之前调用
disable_network()

import transformers
'''
    print("网络隔离方法:")
    print(isolation_code)

def create_offline_hipporag_patch():
    """创建 HippoRAG 的离线补丁"""
    print("\n🔧 创建 HippoRAG 离线补丁")
    print("-" * 30)
    
    patch_content = '''
"""
HippoRAG 离线补丁
在导入 HippoRAG 之前运行此代码
"""

import os
import sys
from unittest.mock import patch, MagicMock

# 设置所有可能的离线环境变量
OFFLINE_ENV_VARS = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'HF_HUB_DISABLE_EXPERIMENTAL_WARNING': '1'
}

for key, value in OFFLINE_ENV_VARS.items():
    os.environ[key] = value

# 猴子补丁网络相关函数
def patch_network_functions():
    """补丁所有可能的网络函数"""
    try:
        import requests
        requests.get = MagicMock(side_effect=Exception("Network disabled"))
        requests.head = MagicMock(side_effect=Exception("Network disabled"))
        requests.post = MagicMock(side_effect=Exception("Network disabled"))
    except ImportError:
        pass
    
    try:
        import urllib.request
        urllib.request.urlopen = MagicMock(side_effect=Exception("Network disabled"))
    except ImportError:
        pass
    
    try:
        import huggingface_hub
        # 禁用 huggingface_hub 的网络请求
        huggingface_hub.hf_hub_download = MagicMock(side_effect=Exception("Network disabled"))
    except ImportError:
        pass

# 应用补丁
patch_network_functions()

print("✅ HippoRAG 离线补丁已应用")
'''
    
    with open('hipporag_offline_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print("已创建离线补丁文件: hipporag_offline_patch.py")
    print("使用方法:")
    print("import hipporag_offline_patch  # 在导入 HippoRAG 之前")
    print("from src.hipporag.HippoRAG import HippoRAG")

def main():
    print("🚀 禁用 transformers 网络验证的方法")
    print("=" * 50)
    
    method1_environment_variables()
    method2_modify_transformers_config()
    method3_patch_huggingface_hub()
    method4_create_offline_wrapper()
    method5_modify_model_config()
    method6_network_isolation()
    create_offline_hipporag_patch()
    
    print("\n🎯 推荐使用顺序:")
    print("1. 首先尝试方法1 (环境变量) - 最简单")
    print("2. 如果不行，使用方法4 (离线包装器)")
    print("3. 最后尝试方法6 (网络隔离) - 最彻底")
    
    print("\n💡 对于您的情况，建议:")
    print("在运行 HippoRAG 之前，先运行:")
    print("export HF_HUB_OFFLINE=1")
    print("export TRANSFORMERS_OFFLINE=1")
    print("或者使用我创建的 hipporag_offline_patch.py")

if __name__ == "__main__":
    main()
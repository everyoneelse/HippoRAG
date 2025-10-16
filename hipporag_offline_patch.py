"""
HippoRAG 完全离线补丁
在导入任何 HippoRAG 模块之前运行此补丁
"""

import os
import sys
from unittest.mock import patch, MagicMock

print("🔧 应用 HippoRAG 离线补丁...")

# 1. 设置所有可能的离线环境变量
OFFLINE_ENV_VARS = {
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1', 
    'HF_HUB_DISABLE_TELEMETRY': '1',
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
    'HF_HUB_DISABLE_EXPERIMENTAL_WARNING': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'CURL_CA_BUNDLE': '',  # 禁用 SSL 验证
    'REQUESTS_CA_BUNDLE': '',
}

for key, value in OFFLINE_ENV_VARS.items():
    os.environ[key] = value
    print(f"  设置 {key}={value}")

# 2. 禁用网络相关模块
def disable_network_modules():
    """禁用所有可能的网络模块"""
    
    # 禁用 requests
    try:
        import requests
        requests.get = MagicMock(side_effect=ConnectionError("Network disabled by offline patch"))
        requests.head = MagicMock(side_effect=ConnectionError("Network disabled by offline patch"))
        requests.post = MagicMock(side_effect=ConnectionError("Network disabled by offline patch"))
        print("  ✅ 已禁用 requests 网络请求")
    except ImportError:
        pass
    
    # 禁用 urllib
    try:
        import urllib.request
        import urllib.error
        
        def mock_urlopen(*args, **kwargs):
            raise urllib.error.URLError("Network disabled by offline patch")
        
        urllib.request.urlopen = mock_urlopen
        print("  ✅ 已禁用 urllib 网络请求")
    except ImportError:
        pass
    
    # 禁用 socket 连接
    try:
        import socket
        original_create_connection = socket.create_connection
        
        def mock_create_connection(address, *args, **kwargs):
            # 只允许本地连接
            if address[0] in ['localhost', '127.0.0.1', '::1']:
                return original_create_connection(address, *args, **kwargs)
            raise ConnectionError(f"Network connection to {address} disabled by offline patch")
        
        socket.create_connection = mock_create_connection
        print("  ✅ 已限制 socket 连接（仅允许本地连接）")
    except ImportError:
        pass

# 3. 禁用 Hugging Face Hub 相关功能
def disable_huggingface_hub():
    """禁用 Hugging Face Hub 网络功能"""
    try:
        import huggingface_hub
        from huggingface_hub import utils
        
        # 禁用下载功能
        huggingface_hub.hf_hub_download = MagicMock(side_effect=ConnectionError("HF Hub disabled"))
        huggingface_hub.snapshot_download = MagicMock(side_effect=ConnectionError("HF Hub disabled"))
        
        # 禁用网络检查
        if hasattr(utils, '_http'):
            utils._http.http_get = MagicMock(side_effect=ConnectionError("HTTP disabled"))
            utils._http.http_head = MagicMock(side_effect=ConnectionError("HTTP disabled"))
        
        print("  ✅ 已禁用 Hugging Face Hub 网络功能")
    except ImportError:
        print("  ⚠️  Hugging Face Hub 未安装，跳过")

# 4. 修改 transformers 的默认行为
def patch_transformers():
    """修改 transformers 的默认行为"""
    try:
        # 在 transformers 加载之前设置
        if 'transformers' not in sys.modules:
            print("  ✅ transformers 尚未加载，预设离线模式")
        else:
            import transformers
            # 如果已经加载，尝试修改配置
            if hasattr(transformers, 'utils'):
                if hasattr(transformers.utils, 'hub'):
                    transformers.utils.hub.HF_HUB_OFFLINE = True
                    print("  ✅ 已设置 transformers 离线模式")
    except Exception as e:
        print(f"  ⚠️  transformers 补丁失败: {e}")

# 5. 创建离线模型加载器
class OfflineModelLoader:
    """完全离线的模型加载器"""
    
    @staticmethod
    def load_model_safely(model_class, model_path, **kwargs):
        """安全的离线模型加载"""
        # 强制离线参数
        offline_kwargs = {
            'local_files_only': True,
            'use_auth_token': False,
            'trust_remote_code': True,
            **kwargs
        }
        
        try:
            print(f"  🔄 尝试离线加载: {model_path}")
            return model_class.from_pretrained(model_path, **offline_kwargs)
        except Exception as e:
            print(f"  ⚠️  离线加载失败: {e}")
            # 尝试最小参数集
            minimal_kwargs = {
                'local_files_only': True,
                'trust_remote_code': True
            }
            print(f"  🔄 尝试最小参数集加载...")
            return model_class.from_pretrained(model_path, **minimal_kwargs)

# 应用所有补丁
print("🔧 应用网络禁用补丁...")
disable_network_modules()
disable_huggingface_hub()
patch_transformers()

# 导出离线加载器
__all__ = ['OfflineModelLoader']

print("✅ HippoRAG 离线补丁应用完成!")
print("📝 现在可以安全导入 HippoRAG 模块")
print("-" * 50)
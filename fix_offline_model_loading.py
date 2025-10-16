#!/usr/bin/env python3
"""
修复本地模型加载时的网络连接问题
"""

import os
import argparse

def set_offline_environment():
    """设置离线环境变量"""
    print("🔧 设置离线环境变量...")
    
    # 设置 Hugging Face 离线模式
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # 禁用网络检查
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    
    print("✅ 已设置以下环境变量:")
    print("   HF_HUB_OFFLINE=1")
    print("   TRANSFORMERS_OFFLINE=1") 
    print("   HF_HUB_DISABLE_TELEMETRY=1")
    print("   HF_HUB_DISABLE_PROGRESS_BARS=1")

def check_local_model_files(model_path):
    """检查本地模型文件是否完整"""
    print(f"\n🔍 检查本地模型文件: {model_path}")
    
    required_files = [
        'config.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'vocab.txt',
        'special_tokens_map.json'
    ]
    
    model_files = [
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model-00001-of-00001.bin'
    ]
    
    missing_files = []
    found_model_file = False
    
    # 检查必需文件
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (缺失)")
            missing_files.append(file)
    
    # 检查模型权重文件（至少需要一个）
    for file in model_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
            found_model_file = True
            break
    
    if not found_model_file:
        print("❌ 未找到模型权重文件 (pytorch_model.bin 或 model.safetensors)")
        missing_files.append("模型权重文件")
    
    if missing_files:
        print(f"\n⚠️  缺失文件: {', '.join(missing_files)}")
        print("💡 建议:")
        print("   1. 重新下载完整的模型文件")
        print("   2. 或者允许网络访问以自动下载缺失文件")
        return False
    else:
        print("\n✅ 所有必需文件都存在")
        return True

def create_offline_demo():
    """创建离线使用示例"""
    demo_content = '''#!/usr/bin/env python3
"""
离线使用 HippoRAG 的示例脚本
"""

import os

# 设置离线环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

def main():
    print("🚀 离线模式 HippoRAG 演示")
    
    # 配置本地模型路径
    local_model_path = "/mnt/data/hy/GPT/models/nvidia/NV-Embed-v2"
    
    config = BaseConfig(
        save_dir='offline_outputs',
        llm_name='gpt-4o-mini',  # 或使用本地 LLM
        embedding_model_name=local_model_path,
        dataset='offline_demo',
        force_index_from_scratch=True
    )
    
    # 初始化 HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # 示例文档
    docs = [
        "人工智能是计算机科学的一个分支。",
        "机器学习是人工智能的核心技术之一。",
        "深度学习基于人工神经网络。"
    ]
    
    # 构建知识图谱
    print("📚 构建知识图谱...")
    hipporag.index(docs)
    
    # 导出知识图谱
    print("💾 导出知识图谱...")
    hipporag.export_knowledge_graph('json', 'offline_graph.json')
    
    print("✅ 离线演示完成!")

if __name__ == "__main__":
    main()
'''
    
    with open('offline_demo.py', 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print("📝 已创建离线演示脚本: offline_demo.py")

def main():
    parser = argparse.ArgumentParser(description="修复本地模型加载问题")
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/data/hy/GPT/models/nvidia/NV-Embed-v2',
                       help='本地模型路径')
    parser.add_argument('--check_files', action='store_true',
                       help='检查本地模型文件完整性')
    parser.add_argument('--set_offline', action='store_true',
                       help='设置离线环境变量')
    parser.add_argument('--create_demo', action='store_true',
                       help='创建离线演示脚本')
    
    args = parser.parse_args()
    
    print("🔧 HippoRAG 本地模型加载修复工具")
    print("=" * 50)
    
    if args.check_files:
        check_local_model_files(args.model_path)
    
    if args.set_offline:
        set_offline_environment()
    
    if args.create_demo:
        create_offline_demo()
    
    if not any([args.check_files, args.set_offline, args.create_demo]):
        # 默认执行所有操作
        check_local_model_files(args.model_path)
        set_offline_environment()
        create_offline_demo()
    
    print(f"\n💡 解决方案总结:")
    print("1. 在运行脚本前设置环境变量:")
    print("   export HF_HUB_OFFLINE=1")
    print("   export TRANSFORMERS_OFFLINE=1")
    
    print(f"\n2. 或者在 Python 代码开头添加:")
    print("   import os")
    print("   os.environ['HF_HUB_OFFLINE'] = '1'")
    print("   os.environ['TRANSFORMERS_OFFLINE'] = '1'")
    
    print(f"\n3. 使用修改后的脚本:")
    print(f"   python export_knowledge_graph.py \\")
    print(f"     --embedding_path {args.model_path} \\")
    print(f"     --dataset sample")

if __name__ == "__main__":
    main()
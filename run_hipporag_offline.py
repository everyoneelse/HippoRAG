#!/usr/bin/env python3
"""
完全离线运行 HippoRAG 的示例脚本
使用离线补丁确保不会有任何网络请求
"""

# 第一步：应用离线补丁（必须在导入其他模块之前）
print("🚀 HippoRAG 完全离线运行")
print("=" * 50)

import hipporag_offline_patch  # 应用离线补丁

# 现在可以安全导入 HippoRAG 相关模块
import os
import argparse
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

def run_offline_demo(model_path, api_key=None):
    """运行离线演示"""
    
    print(f"\n📁 使用本地模型: {model_path}")
    
    # 验证模型路径
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查关键文件
    required_files = ['config.json', 'tokenizer_config.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  警告: 缺失文件 {missing_files}，可能影响加载")
    else:
        print("✅ 模型文件检查通过")
    
    # 设置 API key（如果提供）
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        print("🔑 已设置 API Key")
    
    # 配置 HippoRAG
    config = BaseConfig(
        save_dir='offline_outputs',
        llm_name='gpt-4o-mini',
        embedding_model_name=model_path,  # 使用本地路径
        dataset='offline_test',
        force_index_from_scratch=True,
        force_openie_from_scratch=True
    )
    
    print("\n🔧 初始化 HippoRAG...")
    try:
        hipporag = HippoRAG(global_config=config)
        print("✅ HippoRAG 初始化成功（完全离线）")
    except Exception as e:
        print(f"❌ HippoRAG 初始化失败: {e}")
        return False
    
    # 准备测试数据
    docs = [
        "Python 是一种高级编程语言。",
        "机器学习是人工智能的重要分支。",
        "深度学习使用神经网络进行模式识别。",
        "自然语言处理帮助计算机理解人类语言。",
        "知识图谱用于表示实体之间的关系。"
    ]
    
    print(f"\n📚 开始构建知识图谱（{len(docs)} 个文档）...")
    try:
        hipporag.index(docs)
        print("✅ 知识图谱构建成功")
    except Exception as e:
        print(f"❌ 知识图谱构建失败: {e}")
        return False
    
    # 获取图谱统计信息
    print("\n📊 知识图谱统计:")
    try:
        graph_info = hipporag.get_graph_info()
        for key, value in graph_info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"⚠️  无法获取图谱统计: {e}")
    
    # 导出知识图谱
    print("\n💾 导出知识图谱...")
    try:
        # 导出 JSON 格式
        json_path = hipporag.export_knowledge_graph('json', 'offline_graph.json')
        print(f"✅ JSON 格式导出: {json_path}")
        
        # 导出 OpenIE 结果
        openie_path = hipporag.export_openie_results('offline_openie.json')
        print(f"✅ OpenIE 结果导出: {openie_path}")
        
        # 导出完整知识库
        saved_files = hipporag.save_complete_knowledge_base('offline_complete')
        print(f"✅ 完整知识库导出: {len(saved_files)} 个文件")
        
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        return False
    
    # 测试检索功能
    print("\n🔍 测试检索功能...")
    try:
        queries = ["什么是机器学习？", "深度学习如何工作？"]
        results = hipporag.retrieve(queries, num_to_retrieve=2)
        
        for i, result in enumerate(results):
            print(f"\n查询 {i+1}: {result.question}")
            print("检索结果:")
            for j, doc in enumerate(result.docs):
                print(f"  {j+1}. {doc[:50]}...")
        
        print("✅ 检索功能测试成功")
        
    except Exception as e:
        print(f"❌ 检索功能测试失败: {e}")
        return False
    
    print(f"\n🎉 离线演示完成!")
    print("📂 生成的文件:")
    print("  - offline_graph.json (知识图谱)")
    print("  - offline_openie.json (OpenIE 结果)")
    print("  - offline_complete/ (完整知识库)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="HippoRAG 完全离线运行演示")
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/data/hy/GPT/models/nvidia/NV-Embed-v2',
                       help='本地嵌入模型路径')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API Key（可选）')
    
    args = parser.parse_args()
    
    # 运行离线演示
    success = run_offline_demo(args.model_path, args.api_key)
    
    if success:
        print("\n✅ 所有测试通过！HippoRAG 可以完全离线运行。")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
    
    print("\n💡 提示:")
    print("  - 确保模型文件完整")
    print("  - 使用 --api_key 参数设置 OpenAI API Key（如需要）")
    print("  - 所有操作都在离线模式下完成，无网络请求")

if __name__ == "__main__":
    main()
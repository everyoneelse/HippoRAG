#!/usr/bin/env python3
"""
HippoRAG知识图谱保存功能演示
"""

import os
import argparse
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

def demo_save_knowledge_graph(embedding_model_path=None, openai_api_key=None):
    """演示知识图谱保存功能"""
    
    print("🚀 HippoRAG知识图谱保存功能演示")
    print("=" * 50)
    
    # 设置 API Key
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        print("🔑 使用提供的 OpenAI API Key")
    elif not os.getenv('OPENAI_API_KEY'):
        print("⚠️  未设置 OPENAI_API_KEY，如果使用 OpenAI 模型可能会失败")
        print("💡 提示: 可以通过以下方式设置:")
        print("   方法1: export OPENAI_API_KEY='your-api-key'")
        print("   方法2: python demo_save_kg.py --openai_api_key your-api-key")
    else:
        # 显示已设置的 API Key（部分遮蔽）
        api_key = os.getenv('OPENAI_API_KEY')
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"✅ 已设置 OPENAI_API_KEY: {masked_key}")
    
    # 准备示例数据
    docs = [
        "苹果公司是一家美国科技公司。",
        "iPhone是苹果公司的主要产品。",
        "史蒂夫·乔布斯是苹果公司的创始人之一。",
        "苹果公司总部位于加利福尼亚州库比蒂诺。",
        "Mac电脑也是苹果公司的重要产品线。",
        "苹果公司在全球拥有众多零售店。"
    ]
    
    # 确定嵌入模型路径
    embedding_model_name = embedding_model_path if embedding_model_path else 'nvidia/NV-Embed-v2'
    
    if embedding_model_path:
        print(f"🔧 使用本地嵌入模型路径: {embedding_model_name}")
    
    # 配置HippoRAG
    config = BaseConfig(
        save_dir='demo_outputs',
        llm_name='gpt-4o-mini',
        embedding_model_name=embedding_model_name,
        dataset='demo',
        force_index_from_scratch=True,  # 重新构建
        force_openie_from_scratch=True
    )
    
    print("📝 初始化HippoRAG...")
    hipporag = HippoRAG(global_config=config)
    
    print("🔍 构建知识图谱...")
    hipporag.index(docs)
    
    print("📊 知识图谱统计信息:")
    graph_info = hipporag.get_graph_info()
    for key, value in graph_info.items():
        print(f"  {key}: {value}")
    
    print("\n💾 保存知识图谱...")
    
    # 1. 默认保存（已自动完成）
    print("✅ 1. 默认格式保存完成（pickle格式）")
    
    # 2. 导出为JSON格式
    json_path = hipporag.export_knowledge_graph('json', 'demo_graph.json')
    print(f"✅ 2. JSON格式导出完成: {json_path}")
    
    # 3. 导出为GraphML格式（可用于Gephi等图分析工具）
    try:
        graphml_path = hipporag.export_knowledge_graph('graphml', 'demo_graph.graphml')
        print(f"✅ 3. GraphML格式导出完成: {graphml_path}")
    except Exception as e:
        print(f"⚠️  3. GraphML格式导出跳过: {str(e)}")
    
    # 4. 导出OpenIE结果
    openie_path = hipporag.export_openie_results('demo_openie.json')
    print(f"✅ 4. OpenIE结果导出完成: {openie_path}")
    
    # 5. 导出完整知识库
    print("\n🎯 导出完整知识库...")
    saved_files = hipporag.save_complete_knowledge_base('complete_export')
    print("✅ 5. 完整知识库导出完成!")
    
    print(f"\n📁 导出的文件:")
    for file_type, file_path in saved_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  {file_type}: {file_path} ({file_size:.1f} KB)")
    
    print(f"\n🎉 演示完成!")
    print(f"📂 您可以在以下位置找到保存的文件:")
    print(f"  - 工作目录: {hipporag.working_dir}")
    print(f"  - 导出目录: complete_export/")
    
    # 演示如何重新加载知识图谱
    print(f"\n🔄 演示重新加载知识图谱...")
    
    # 创建新的HippoRAG实例（使用相同配置）
    config_reload = BaseConfig(
        save_dir='demo_outputs',
        llm_name='gpt-4o-mini',
        embedding_model_name=embedding_model_name,  # 使用相同的嵌入模型路径
        dataset='demo',
        force_index_from_scratch=False,  # 使用已保存的数据
        force_openie_from_scratch=False
    )
    
    hipporag_reloaded = HippoRAG(global_config=config_reload)
    
    # 测试检索功能
    queries = ["苹果公司的创始人是谁？", "iPhone是什么产品？"]
    
    print("🔍 测试检索功能...")
    retrieval_results = hipporag_reloaded.retrieve(queries=queries, num_to_retrieve=2)
    
    for i, result in enumerate(retrieval_results):
        print(f"\n查询 {i+1}: {result.question}")
        print("检索结果:")
        for j, doc in enumerate(result.docs):
            print(f"  {j+1}. {doc[:100]}...")
    
    print(f"\n✅ 知识图谱重新加载和检索测试成功!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HippoRAG知识图谱保存功能演示")
    parser.add_argument('--embedding_path', type=str, default=None, 
                       help='本地嵌入模型路径（如果模型下载在本地）')
    parser.add_argument('--openai_api_key', type=str, default=None,
                       help='OpenAI API Key（也可通过环境变量 OPENAI_API_KEY 设置）')
    
    args = parser.parse_args()
    
    demo_save_knowledge_graph(args.embedding_path, args.openai_api_key)
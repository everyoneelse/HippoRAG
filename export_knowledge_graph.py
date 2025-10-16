#!/usr/bin/env python3
"""
知识图谱导出示例脚本
演示如何使用HippoRAG的各种知识图谱保存和导出功能
"""

import os
import json
import argparse
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig

def main():
    parser = argparse.ArgumentParser(description="导出HippoRAG知识图谱")
    parser.add_argument('--dataset', type=str, default='sample', help='数据集名称')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM模型名称')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='嵌入模型名称')
    parser.add_argument('--embedding_path', type=str, default=None, help='本地嵌入模型路径（如果模型下载在本地）')
    parser.add_argument('--save_dir', type=str, default='outputs', help='保存目录')
    parser.add_argument('--export_format', type=str, choices=['json', 'graphml', 'gml', 'all'], 
                       default='all', help='导出格式')
    parser.add_argument('--export_dir', type=str, default='knowledge_exports', help='导出目录')
    parser.add_argument('--openai_api_key', type=str, default=None, 
                       help='OpenAI API Key（也可通过环境变量 OPENAI_API_KEY 设置）')
    parser.add_argument('--llm_base_url', type=str, default=None,
                       help='LLM服务的base URL（用于本地部署的模型）')
    parser.add_argument('--offline', action='store_true',
                       help='离线模式，不尝试连接网络下载模型文件')
    
    args = parser.parse_args()
    
    # 设置离线模式
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        print("🔧 已启用离线模式，将不会尝试网络连接")
    
    # 设置 API Key
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
        print("🔑 使用命令行提供的 OpenAI API Key")
    elif not os.getenv('OPENAI_API_KEY'):
        print("⚠️  未设置 OPENAI_API_KEY 环境变量")
        if 'gpt' in args.llm_name.lower() or 'openai' in args.llm_name.lower():
            print("💡 提示: 使用 OpenAI 模型需要设置 API Key")
            print("   方法1: export OPENAI_API_KEY='your-api-key'")
            print("   方法2: --openai_api_key your-api-key")
    else:
        # 显示已设置的 API Key（部分遮蔽）
        api_key = os.getenv('OPENAI_API_KEY')
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"✅ 已设置 OPENAI_API_KEY: {masked_key}")
    
    # 设置路径
    dataset_name = args.dataset
    save_dir = os.path.join(args.save_dir, dataset_name)
    export_dir = args.export_dir
    
    # 处理嵌入模型路径
    embedding_model_name = args.embedding_name
    if args.embedding_path:
        # 如果指定了本地路径，使用本地路径
        embedding_model_name = args.embedding_path
        print(f"🔧 使用本地嵌入模型路径: {embedding_model_name}")
    
    # 确保导出目录存在
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"🚀 开始导出知识图谱...")
    print(f"📁 数据集: {dataset_name}")
    print(f"🤖 LLM模型: {args.llm_name}")
    print(f"🔢 嵌入模型: {embedding_model_name}")
    print(f"📂 导出目录: {export_dir}")
    
    # 配置HippoRAG
    config = BaseConfig(
        save_dir=save_dir,
        llm_name=args.llm_name,
        llm_base_url=args.llm_base_url,  # 支持自定义 LLM base URL
        embedding_model_name=embedding_model_name,  # 使用处理后的嵌入模型名称/路径
        dataset=dataset_name,
        force_index_from_scratch=False,  # 使用已有的索引
        force_openie_from_scratch=False
    )
    
    # 初始化HippoRAG实例
    hipporag = HippoRAG(global_config=config)
    
    # 如果没有现有的知识图谱，先创建一个简单的示例
    if not os.path.exists(hipporag._graph_pickle_filename):
        print("⚠️  未找到现有知识图谱，创建示例数据...")
        
        # 示例文档
        docs = [
            "北京是中国的首都。",
            "中国是一个拥有悠久历史的国家。",
            "长城是中国著名的古建筑。",
            "故宫位于北京市中心。",
            "天安门广场是世界上最大的城市广场之一。"
        ]
        
        # 索引文档
        hipporag.index(docs)
        print("✅ 示例知识图谱创建完成！")
    
    # 准备检索对象
    if not hipporag.ready_to_retrieve:
        hipporag.prepare_retrieval_objects()
    
    # 导出知识图谱
    print("\n📊 知识图谱统计信息:")
    graph_info = hipporag.get_graph_info()
    for key, value in graph_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n💾 开始导出知识图谱...")
    
    exported_files = []
    
    if args.export_format == 'all':
        # 导出所有格式
        formats = ['json', 'graphml', 'gml']
        for fmt in formats:
            try:
                output_path = os.path.join(export_dir, f"knowledge_graph_{dataset_name}.{fmt}")
                exported_path = hipporag.export_knowledge_graph(fmt, output_path)
                exported_files.append(exported_path)
                print(f"✅ {fmt.upper()}格式导出完成: {exported_path}")
            except Exception as e:
                print(f"❌ {fmt.upper()}格式导出失败: {str(e)}")
    else:
        # 导出指定格式
        try:
            output_path = os.path.join(export_dir, f"knowledge_graph_{dataset_name}.{args.export_format}")
            exported_path = hipporag.export_knowledge_graph(args.export_format, output_path)
            exported_files.append(exported_path)
            print(f"✅ {args.export_format.upper()}格式导出完成: {exported_path}")
        except Exception as e:
            print(f"❌ {args.export_format.upper()}格式导出失败: {str(e)}")
    
    # 导出OpenIE结果
    try:
        openie_path = os.path.join(export_dir, f"openie_results_{dataset_name}.json")
        exported_path = hipporag.export_openie_results(openie_path)
        exported_files.append(exported_path)
        print(f"✅ OpenIE结果导出完成: {exported_path}")
    except Exception as e:
        print(f"❌ OpenIE结果导出失败: {str(e)}")
    
    # 导出完整知识库
    try:
        print(f"\n🎯 导出完整知识库...")
        saved_files = hipporag.save_complete_knowledge_base(export_dir)
        print(f"✅ 完整知识库导出完成!")
        print(f"📋 导出摘要: {saved_files['summary']}")
        
        # 显示摘要信息
        with open(saved_files['summary'], 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"\n📈 导出摘要:")
        print(f"  导出时间: {summary['export_timestamp']}")
        print(f"  导出目录: {summary['export_directory']}")
        print(f"  LLM模型: {summary['configuration']['llm_model']}")
        print(f"  嵌入模型: {summary['configuration']['embedding_model']}")
        print(f"  数据集: {summary['configuration']['dataset']}")
        
        print(f"\n📁 导出的文件:")
        for file_type, file_path in saved_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {file_type}: {file_path} ({file_size:.1f} KB)")
        
    except Exception as e:
        print(f"❌ 完整知识库导出失败: {str(e)}")
    
    print(f"\n🎉 知识图谱导出完成!")
    print(f"📂 所有文件已保存到: {export_dir}")

if __name__ == "__main__":
    main()
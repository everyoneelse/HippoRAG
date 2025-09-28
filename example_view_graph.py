#!/usr/bin/env python3
"""
HippoRAG图谱查看示例

这个示例展示了如何在构建HippoRAG图谱后查看和分析图谱结构
"""

from src.hipporag.HippoRAG import HippoRAG

def example_build_and_view_graph():
    """示例：构建并查看图谱"""
    
    print("🚀 HippoRAG 图谱构建与查看示例")
    print("=" * 60)
    
    # 准备示例数据
    docs = [
        "Oliver Badman is a politician from the UK.",
        "George Rankin is a politician from Scotland.", 
        "Thomas Marwick is a politician from Ireland.",
        "Cinderella attended the royal ball in the kingdom.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello in California.",
        "Marina was born in Minsk, the capital of Belarus.",
        "Montebello is a part of Rockland County in New York.",
        "John Smith is a researcher at Stanford University.",
        "Stanford University is located in California, United States.",
        "California is known for its technology companies.",
        "Apple Inc. was founded by Steve Jobs in California.",
        "Steve Jobs was a visionary entrepreneur and inventor."
    ]
    
    # 配置参数
    save_dir = 'outputs'
    llm_model_name = 'gpt-4o-mini'  # 使用OpenAI模型
    embedding_model_name = 'nvidia/NV-Embed-v2'
    
    print(f"📁 保存目录: {save_dir}")
    print(f"🤖 LLM模型: {llm_model_name}")
    print(f"📊 嵌入模型: {embedding_model_name}")
    print(f"📄 文档数量: {len(docs)}")
    
    try:
        # 1. 初始化HippoRAG
        print("\n🔧 初始化HippoRAG...")
        hipporag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name
        )
        
        # 2. 构建图谱
        print("\n🏗️  开始构建知识图谱...")
        hipporag.index(docs=docs)
        print("✅ 图谱构建完成！")
        
        # 3. 查看图谱统计信息
        print("\n📊 知识图谱统计信息:")
        print("=" * 40)
        graph_info = hipporag.get_graph_info()
        
        print(f"🔸 实体节点数量: {graph_info['num_phrase_nodes']:,}")
        print(f"🔸 文档节点数量: {graph_info['num_passage_nodes']:,}")
        print(f"🔸 总节点数量: {graph_info['num_total_nodes']:,}")
        print(f"🔸 提取的三元组: {graph_info['num_extracted_triples']:,}")
        print(f"🔸 包含文档节点的关系: {graph_info['num_triples_with_passage_node']:,}")
        print(f"🔸 同义词关系: {graph_info['num_synonymy_triples']:,}")
        print(f"🔸 总关系数量: {graph_info['num_total_triples']:,}")
        
        # 4. 查看实体节点
        print(f"\n🏷️  实体节点示例:")
        print("-" * 30)
        entity_nodes = hipporag.entity_embedding_store.get_all_ids()
        for i, node in enumerate(entity_nodes[:10], 1):
            print(f"  {i:2d}. {node}")
        
        if len(entity_nodes) > 10:
            print(f"  ... 还有 {len(entity_nodes) - 10} 个实体节点")
        
        # 5. 查看文档节点
        print(f"\n📄 文档节点示例:")
        print("-" * 30)
        passage_nodes = hipporag.chunk_embedding_store.get_all_ids()
        for i, node in enumerate(passage_nodes[:5], 1):
            # 截断长文档以便显示
            display_node = node[:60] + "..." if len(node) > 60 else node
            print(f"  {i}. {display_node}")
        
        if len(passage_nodes) > 5:
            print(f"  ... 还有 {len(passage_nodes) - 5} 个文档节点")
        
        # 6. 查看提取的三元组
        print(f"\n🔗 提取的三元组示例:")
        print("-" * 30)
        fact_ids = hipporag.fact_embedding_store.get_all_ids()
        for i, fact in enumerate(fact_ids[:8], 1):
            print(f"  {i}. {fact}")
        
        if len(fact_ids) > 8:
            print(f"  ... 还有 {len(fact_ids) - 8} 个三元组")
        
        # 7. 测试检索功能
        print(f"\n🔍 测试图谱检索功能:")
        print("-" * 30)
        test_queries = [
            "Who is a politician?",
            "Where is Montebello located?", 
            "What happened to Cinderella?",
            "Who founded Apple Inc.?"
        ]
        
        for query in test_queries:
            print(f"\n❓ 查询: {query}")
            try:
                # 执行检索
                results = hipporag.retrieve(queries=[query], num_to_retrieve=3)
                print(f"✅ 检索到 {len(results[0])} 个相关文档")
                
                # 显示检索结果
                for j, result in enumerate(results[0][:2], 1):
                    content = result[:80] + "..." if len(result) > 80 else result
                    print(f"  {j}. {content}")
                    
            except Exception as e:
                print(f"❌ 检索失败: {e}")
        
        print(f"\n" + "=" * 60)
        print("✅ 图谱构建与查看示例完成！")
        
        # 8. 提供后续操作建议
        print(f"\n💡 后续操作建议:")
        print(f"  1. 使用以下命令查看详细图谱信息:")
        print(f"     python simple_graph_viewer.py {save_dir} {llm_model_name} {embedding_model_name}")
        print(f"")
        print(f"  2. 使用以下命令进行高级查看和可视化:")
        print(f"     python visualize_graph.py --save_dir {save_dir} --llm_name {llm_model_name} --embedding_name {embedding_model_name} --export")
        print(f"")
        print(f"  3. 搜索特定节点:")
        print(f"     python visualize_graph.py --save_dir {save_dir} --llm_name {llm_model_name} --embedding_name {embedding_model_name} --search politician")
        
        return hipporag
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        print(f"请检查:")
        print(f"  - 是否设置了 OPENAI_API_KEY 环境变量")
        print(f"  - 网络连接是否正常")
        print(f"  - 依赖包是否正确安装")
        return None

if __name__ == "__main__":
    # 运行示例
    example_build_and_view_graph()
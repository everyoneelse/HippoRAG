#!/usr/bin/env python3
"""
简单的HippoRAG图谱查看器

快速查看已构建的HippoRAG知识图谱的基本信息
"""

import os
import sys
from src.hipporag.HippoRAG import HippoRAG

def view_graph_simple(save_dir, llm_name, embedding_name):
    """简单查看图谱信息"""
    
    print("🔍 正在加载HippoRAG实例...")
    
    try:
        # 初始化HippoRAG实例
        hipporag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=llm_name,
            embedding_model_name=embedding_name
        )
        
        print("✅ HippoRAG实例加载成功！")
        print("=" * 60)
        
        # 获取并显示图谱统计信息
        graph_info = hipporag.get_graph_info()
        
        print("📊 知识图谱统计信息:")
        print(f"  🔸 实体节点: {graph_info['num_phrase_nodes']:,} 个")
        print(f"  🔸 文档节点: {graph_info['num_passage_nodes']:,} 个") 
        print(f"  🔸 总节点数: {graph_info['num_total_nodes']:,} 个")
        print(f"  🔸 提取三元组: {graph_info['num_extracted_triples']:,} 个")
        print(f"  🔸 文档关系: {graph_info['num_triples_with_passage_node']:,} 个")
        print(f"  🔸 同义关系: {graph_info['num_synonymy_triples']:,} 个")
        print(f"  🔸 总关系数: {graph_info['num_total_triples']:,} 个")
        
        print("\n" + "=" * 60)
        
        # 显示一些示例节点
        print("🔸 实体节点示例:")
        entity_nodes = hipporag.entity_embedding_store.get_all_ids()
        for i, node in enumerate(entity_nodes[:5], 1):
            print(f"  {i}. {node}")
        
        if len(entity_nodes) > 5:
            print(f"  ... 还有 {len(entity_nodes) - 5} 个实体节点")
        
        print("\n🔸 文档节点示例:")
        passage_nodes = hipporag.chunk_embedding_store.get_all_ids()
        for i, node in enumerate(passage_nodes[:3], 1):
            # 截断长文档名称以便显示
            display_name = node[:80] + "..." if len(node) > 80 else node
            print(f"  {i}. {display_name}")
        
        if len(passage_nodes) > 3:
            print(f"  ... 还有 {len(passage_nodes) - 3} 个文档节点")
        
        print("\n🔸 三元组示例:")
        fact_ids = hipporag.fact_embedding_store.get_all_ids()
        for i, fact in enumerate(fact_ids[:5], 1):
            print(f"  {i}. {fact}")
        
        if len(fact_ids) > 5:
            print(f"  ... 还有 {len(fact_ids) - 5} 个三元组")
        
        print("\n" + "=" * 60)
        print("✅ 图谱信息显示完成！")
        
        # 提供更多操作选项
        print("\n💡 更多操作选项:")
        print("  - 使用 visualize_graph.py 进行详细查看和可视化")
        print("  - 使用 --search 参数搜索特定节点")
        print("  - 使用 --export 参数导出图谱数据")
        
        return hipporag
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保指定的路径存在且包含已构建的HippoRAG数据")
        return None
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请检查参数是否正确")
        return None

def main():
    """主函数"""
    if len(sys.argv) != 4:
        print("用法: python simple_graph_viewer.py <save_dir> <llm_name> <embedding_name>")
        print("")
        print("示例:")
        print("  python simple_graph_viewer.py outputs gpt-4o-mini nvidia/NV-Embed-v2")
        print("  python simple_graph_viewer.py outputs meta-llama/Llama-3.3-70B-Instruct nvidia/NV-Embed-v2")
        sys.exit(1)
    
    save_dir = sys.argv[1]
    llm_name = sys.argv[2] 
    embedding_name = sys.argv[3]
    
    print("🚀 HippoRAG 简单图谱查看器")
    print(f"📁 保存目录: {save_dir}")
    print(f"🤖 LLM模型: {llm_name}")
    print(f"📊 嵌入模型: {embedding_name}")
    print("=" * 60)
    
    view_graph_simple(save_dir, llm_name, embedding_name)

if __name__ == "__main__":
    main()
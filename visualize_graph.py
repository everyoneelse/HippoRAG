#!/usr/bin/env python3
"""
HippoRAG图谱可视化工具

这个脚本展示了如何查看和可视化HippoRAG系统构建的知识图谱。
支持多种查看方式：
1. 图谱统计信息
2. 节点和边的详细信息
3. 图谱可视化（如果安装了可视化库）
4. 导出图谱数据
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("提示: 安装 matplotlib 和 networkx 以启用图谱可视化功能")
    print("运行: pip install matplotlib networkx")

from src.hipporag.HippoRAG import HippoRAG

class GraphViewer:
    """HippoRAG图谱查看器"""
    
    def __init__(self, hipporag_instance: HippoRAG):
        """
        初始化图谱查看器
        
        Args:
            hipporag_instance: HippoRAG实例
        """
        self.hipporag = hipporag_instance
        self.graph = hipporag_instance.graph
        
    def show_graph_statistics(self) -> Dict:
        """显示图谱统计信息"""
        print("=" * 60)
        print("📊 HippoRAG 知识图谱统计信息")
        print("=" * 60)
        
        # 获取图谱信息
        graph_info = self.hipporag.get_graph_info()
        
        print(f"🔹 节点统计:")
        print(f"  - 实体节点数量: {graph_info['num_phrase_nodes']:,}")
        print(f"  - 文档节点数量: {graph_info['num_passage_nodes']:,}")
        print(f"  - 总节点数量: {graph_info['num_total_nodes']:,}")
        
        print(f"\n🔹 关系统计:")
        print(f"  - 提取的三元组: {graph_info['num_extracted_triples']:,}")
        print(f"  - 包含文档节点的三元组: {graph_info['num_triples_with_passage_node']:,}")
        print(f"  - 同义词关系: {graph_info['num_synonymy_triples']:,}")
        print(f"  - 总关系数量: {graph_info['num_total_triples']:,}")
        
        # 图谱密度
        if graph_info['num_total_nodes'] > 1:
            max_edges = graph_info['num_total_nodes'] * (graph_info['num_total_nodes'] - 1)
            if not self.hipporag.global_config.is_directed_graph:
                max_edges //= 2
            density = graph_info['num_total_triples'] / max_edges if max_edges > 0 else 0
            print(f"\n🔹 图谱密度: {density:.4f}")
        
        print("=" * 60)
        return graph_info
    
    def show_nodes_info(self, limit: int = 10) -> None:
        """显示节点信息"""
        print("\n📋 节点信息预览")
        print("-" * 40)
        
        # 显示实体节点
        entity_nodes = self.hipporag.entity_embedding_store.get_all_ids()[:limit]
        print(f"🔸 实体节点 (前{min(limit, len(entity_nodes))}个):")
        for i, node in enumerate(entity_nodes, 1):
            print(f"  {i}. {node}")
        
        # 显示文档节点
        passage_nodes = self.hipporag.chunk_embedding_store.get_all_ids()[:limit]
        print(f"\n🔸 文档节点 (前{min(limit, len(passage_nodes))}个):")
        for i, node in enumerate(passage_nodes, 1):
            # 截断长文档名称
            display_name = node[:50] + "..." if len(node) > 50 else node
            print(f"  {i}. {display_name}")
    
    def show_edges_info(self, limit: int = 10) -> None:
        """显示边信息"""
        print(f"\n🔗 关系信息预览")
        print("-" * 40)
        
        # 显示提取的三元组
        fact_ids = self.hipporag.fact_embedding_store.get_all_ids()[:limit]
        print(f"🔸 提取的三元组 (前{min(limit, len(fact_ids))}个):")
        for i, fact_id in enumerate(fact_ids, 1):
            # 解析三元组
            try:
                # 假设fact_id是三元组的字符串表示
                print(f"  {i}. {fact_id}")
            except:
                print(f"  {i}. [复杂关系]")
        
        # 显示节点间连接统计
        if hasattr(self.hipporag, 'node_to_node_stats'):
            print(f"\n🔸 节点连接统计:")
            node_connections = list(self.hipporag.node_to_node_stats.items())[:limit]
            for i, ((node1, node2), weight) in enumerate(node_connections, 1):
                node1_short = node1[:20] + "..." if len(node1) > 20 else node1
                node2_short = node2[:20] + "..." if len(node2) > 20 else node2
                print(f"  {i}. {node1_short} ↔ {node2_short} (权重: {weight:.3f})")
    
    def search_nodes(self, keyword: str, limit: int = 10) -> List[str]:
        """搜索包含关键词的节点"""
        print(f"\n🔍 搜索包含 '{keyword}' 的节点")
        print("-" * 40)
        
        found_nodes = []
        
        # 搜索实体节点
        entity_nodes = self.hipporag.entity_embedding_store.get_all_ids()
        matching_entities = [node for node in entity_nodes if keyword.lower() in node.lower()]
        
        # 搜索文档节点
        passage_nodes = self.hipporag.chunk_embedding_store.get_all_ids()
        matching_passages = [node for node in passage_nodes if keyword.lower() in node.lower()]
        
        print(f"🔸 匹配的实体节点 ({len(matching_entities)}个):")
        for i, node in enumerate(matching_entities[:limit], 1):
            print(f"  {i}. {node}")
            found_nodes.append(node)
        
        print(f"\n🔸 匹配的文档节点 ({len(matching_passages)}个):")
        for i, node in enumerate(matching_passages[:limit], 1):
            display_name = node[:60] + "..." if len(node) > 60 else node
            print(f"  {i}. {display_name}")
            found_nodes.append(node)
        
        return found_nodes
    
    def export_graph_data(self, output_dir: str = "graph_export") -> None:
        """导出图谱数据"""
        print(f"\n💾 导出图谱数据到 {output_dir}/")
        print("-" * 40)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出节点信息
        nodes_data = {
            'entity_nodes': self.hipporag.entity_embedding_store.get_all_ids(),
            'passage_nodes': self.hipporag.chunk_embedding_store.get_all_ids()
        }
        
        with open(os.path.join(output_dir, 'nodes.json'), 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        print("✓ 节点数据已导出到 nodes.json")
        
        # 导出边信息
        edges_data = {
            'facts': self.hipporag.fact_embedding_store.get_all_ids(),
            'node_connections': dict(self.hipporag.node_to_node_stats) if hasattr(self.hipporag, 'node_to_node_stats') else {}
        }
        
        with open(os.path.join(output_dir, 'edges.json'), 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        print("✓ 边数据已导出到 edges.json")
        
        # 导出图谱统计
        graph_stats = self.hipporag.get_graph_info()
        with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(graph_stats, f, ensure_ascii=False, indent=2)
        print("✓ 统计信息已导出到 statistics.json")
        
        print(f"📁 所有数据已导出到 {output_dir}/ 目录")
    
    def visualize_subgraph(self, center_nodes: List[str], max_nodes: int = 50, save_path: Optional[str] = None) -> None:
        """可视化子图"""
        if not HAS_VISUALIZATION:
            print("❌ 需要安装 matplotlib 和 networkx 才能使用可视化功能")
            print("运行: pip install matplotlib networkx")
            return
        
        print(f"\n🎨 生成子图可视化 (最多{max_nodes}个节点)")
        print("-" * 40)
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 添加中心节点
        for node in center_nodes:
            G.add_node(node, node_type='center')
        
        # 添加相关节点和边
        added_nodes = set(center_nodes)
        
        if hasattr(self.hipporag, 'node_to_node_stats'):
            for (node1, node2), weight in self.hipporag.node_to_node_stats.items():
                if len(added_nodes) >= max_nodes:
                    break
                
                if node1 in center_nodes or node2 in center_nodes:
                    G.add_node(node1, node_type='entity' if node1 not in self.hipporag.chunk_embedding_store.get_all_ids() else 'passage')
                    G.add_node(node2, node_type='entity' if node2 not in self.hipporag.chunk_embedding_store.get_all_ids() else 'passage')
                    G.add_edge(node1, node2, weight=weight)
                    added_nodes.add(node1)
                    added_nodes.add(node2)
        
        if len(G.nodes()) == 0:
            print("❌ 没有找到相关的图结构数据")
            return
        
        # 设置图形大小
        plt.figure(figsize=(12, 8))
        
        # 计算布局
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制不同类型的节点
        center_nodes_in_graph = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'center']
        entity_nodes_in_graph = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'entity']
        passage_nodes_in_graph = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'passage']
        
        # 绘制节点
        if center_nodes_in_graph:
            nx.draw_networkx_nodes(G, pos, nodelist=center_nodes_in_graph, 
                                 node_color='red', node_size=300, alpha=0.8, label='中心节点')
        if entity_nodes_in_graph:
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes_in_graph, 
                                 node_color='lightblue', node_size=200, alpha=0.7, label='实体节点')
        if passage_nodes_in_graph:
            nx.draw_networkx_nodes(G, pos, nodelist=passage_nodes_in_graph, 
                                 node_color='lightgreen', node_size=250, alpha=0.7, label='文档节点')
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
        
        # 添加标签（只为中心节点）
        center_labels = {n: n[:15] + "..." if len(n) > 15 else n for n in center_nodes_in_graph}
        nx.draw_networkx_labels(G, pos, labels=center_labels, font_size=8)
        
        plt.title(f"HippoRAG 知识图谱子图\n中心节点: {', '.join([n[:20] for n in center_nodes])}")
        plt.legend()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化图片已保存到 {save_path}")
        else:
            plt.show()
        
        print(f"✓ 子图包含 {len(G.nodes())} 个节点，{len(G.edges())} 条边")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HippoRAG图谱查看工具")
    parser.add_argument("--save_dir", type=str, required=True, help="HippoRAG保存目录")
    parser.add_argument("--llm_name", type=str, required=True, help="LLM模型名称")
    parser.add_argument("--embedding_name", type=str, required=True, help="嵌入模型名称")
    parser.add_argument("--search", type=str, help="搜索关键词")
    parser.add_argument("--visualize", type=str, help="可视化中心节点（用逗号分隔多个节点）")
    parser.add_argument("--export", action="store_true", help="导出图谱数据")
    parser.add_argument("--limit", type=int, default=10, help="显示结果限制数量")
    
    args = parser.parse_args()
    
    print("🚀 启动 HippoRAG 图谱查看器")
    print("=" * 60)
    
    try:
        # 初始化HippoRAG实例
        hipporag = HippoRAG(
            save_dir=args.save_dir,
            llm_model_name=args.llm_name,
            embedding_model_name=args.embedding_name
        )
        
        # 创建图谱查看器
        viewer = GraphViewer(hipporag)
        
        # 显示基本统计信息
        viewer.show_graph_statistics()
        
        # 显示节点和边信息
        viewer.show_nodes_info(limit=args.limit)
        viewer.show_edges_info(limit=args.limit)
        
        # 搜索功能
        if args.search:
            viewer.search_nodes(args.search, limit=args.limit)
        
        # 可视化功能
        if args.visualize:
            center_nodes = [node.strip() for node in args.visualize.split(',')]
            viewer.visualize_subgraph(center_nodes, max_nodes=50, save_path="graph_visualization.png")
        
        # 导出功能
        if args.export:
            viewer.export_graph_data()
        
        print(f"\n✅ 图谱查看完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print(f"请确保HippoRAG实例已经正确初始化并构建了图谱")


if __name__ == "__main__":
    main()
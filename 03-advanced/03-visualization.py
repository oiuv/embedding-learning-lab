#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级功能第3课：高级可视化系统
=========================

基于文本嵌入的高级数据可视化系统，实现3D空间展示、交互式图表、时间序列分析等。

学习目标：
1. 构建3D空间可视化
2. 实现交互式可视化界面
3. 时间序列嵌入分析
4. 聚类结果动态展示
5. 语义网络图可视化
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
from sklearn.decomposition import PCA

# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
from sklearn.manifold import TSNE
import networkx as nx
from collections import defaultdict
import seaborn as sns

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

class AdvancedVisualizationSystem:
    """高级可视化系统"""
    
    def __init__(self):
        """初始化可视化系统"""
        self.client = EmbeddingClient()
        self.color_palettes = {
            'tech': '#1f77b4',
            'sports': '#ff7f0e', 
            'finance': '#2ca02c',
            'entertainment': '#d62728',
            'default': '#9467bd'
        }
        
    def load_sample_data(self) -> Dict[str, List[str]]:
        """加载示例数据"""
        # 包含时间戳的数据
        base_date = datetime.now()
        sample_data = {
            '科技新闻': [
                {'text': '人工智能技术突破，深度学习模型参数量突破万亿级别', 'date': base_date - timedelta(days=1)},
                {'text': '量子计算机实现量子霸权，计算速度提升万倍', 'date': base_date - timedelta(days=3)},
                {'text': '5G网络全球部署完成，物联网设备连接数超过100亿', 'date': base_date - timedelta(days=5)},
                {'text': '自动驾驶技术重大突破，L4级别车辆开始商业化运营', 'date': base_date - timedelta(days=7)},
                {'text': '区块链技术在金融领域应用扩大，数字货币交易量创新高', 'date': base_date - timedelta(days=9)}
            ],
            '体育新闻': [
                {'text': '国足世界杯预选赛取得胜利，球迷热情高涨', 'date': base_date - timedelta(days=2)},
                {'text': 'NBA总决赛湖人vs凯尔特人抢七大战即将开始', 'date': base_date - timedelta(days=4)},
                {'text': '中国女排世界锦标赛夺冠，展现强大实力', 'date': base_date - timedelta(days=6)},
                {'text': '奥运会筹备工作完成，各国运动员陆续抵达', 'date': base_date - timedelta(days=8)},
                {'text': '世界杯足球赛分组抽签结果公布，强队云集', 'date': base_date - timedelta(days=10)}
            ],
            '财经新闻': [
                {'text': '央行宣布降息政策，刺激经济增长', 'date': base_date - timedelta(days=1)},
                {'text': '股市今日大涨，科技股领涨大盘', 'date': base_date - timedelta(days=2)},
                {'text': '房地产市场调控政策效果显著，房价稳定', 'date': base_date - timedelta(days=4)},
                {'text': '国际贸易额创新高，双边合作加强', 'date': base_date - timedelta(days=6)},
                {'text': '数字货币试点扩大，金融科技发展迅速', 'date': base_date - timedelta(days=8)}
            ]
        }
        return sample_data
    
    def create_3d_visualization(self, texts: List[str], labels: List[str] = None) -> Dict:
        """创建3D空间可视化"""
        print("🎯 创建3D空间可视化...")
        
        # 获取嵌入向量
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 使用PCA降维到3D
        pca_3d = PCA(n_components=3)
        embeddings_3d = pca_3d.fit_transform(embeddings_array)
        
        # 创建3D图形
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 根据标签着色
        if labels:
            unique_labels = list(set(labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                          c=[colors[i]], label=label, alpha=0.7, s=50)
        else:
            ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                      c='blue', alpha=0.6, s=50)
        
        ax.set_xlabel('主成分1', fontsize=12)
        ax.set_ylabel('主成分2', fontsize=12)
        ax.set_zlabel('主成分3', fontsize=12)
        ax.set_title('文本嵌入3D空间可视化', fontsize=14)
        ax.legend()
        
        filename = "03-advanced/3d_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'embeddings_3d': embeddings_3d,
            'explained_variance_ratio': pca_3d.explained_variance_ratio_,
            'filename': filename
        }
    
    def create_interactive_plotly(self, texts: List[str], categories: List[str] = None) -> str:
        """创建交互式Plotly可视化"""
        print("🎯 创建交互式可视化...")
        
        # 获取嵌入向量
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'text': texts,
            'category': categories or ['default'] * len(texts),
            'length': [len(text) for text in texts]
        })
        
        # 创建交互式图形
        fig = px.scatter(df, x='x', y='y', 
                        color='category',
                        text='text',
                        size='length',
                        hover_data={'text': True, 'category': True},
                        title='文本嵌入交互式可视化')
        
        # 更新布局
        fig.update_traces(textposition='top center', textfont_size=8)
        fig.update_layout(
            height=600,
            showlegend=True,
            xaxis_title="t-SNE维度1",
            yaxis_title="t-SNE维度2"
        )
        
        filename = "03-advanced/interactive_visualization.html"
        fig.write_html(filename)
        # 在命令行环境不显示，只保存文件
        # fig.show()
        
        return filename
    
    def create_time_series_analysis(self, timed_texts: List[Dict]) -> Dict:
        """创建时间序列嵌入分析"""
        print("🎯 创建时间序列分析...")
        
        # 处理时间序列数据
        categories = list(set([item['category'] for item in timed_texts]))
        
        # 为每个类别创建时间序列分析
        results = {}
        
        for category in categories:
            category_data = [item for item in timed_texts if item['category'] == category]
            
            # 按时间排序
            category_data.sort(key=lambda x: x['date'])
            
            # 获取嵌入向量
            texts = [item['text'] for item in category_data]
            embeddings = [self.client.get_embedding(text) for text in texts]
            embeddings_array = np.array(embeddings)
            
            # 使用PCA降维到1D（时间轴）
            pca_1d = PCA(n_components=1)
            time_embeddings = pca_1d.fit_transform(embeddings_array)
            
            # 创建时间序列图
            dates = [item['date'] for item in category_data]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, time_embeddings.flatten(), marker='o', linewidth=2, markersize=6)
            plt.title(f'{category} - 语义演进时间序列', fontsize=14)
            plt.xlabel('时间', fontsize=12)
            plt.ylabel('语义位置（主成分1）', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加文本标注
            for i, (date, text, embedding) in enumerate(zip(dates, texts, time_embeddings)):
                if i % 2 == 0:  # 每隔一个显示
                    plt.annotate(text[:20] + '...', 
                               (date, embedding[0]), 
                               xytext=(10, 10), 
                               textcoords='offset points',
                               fontsize=8, alpha=0.7)
            
            filename = f"03-advanced/time_series_{category}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            results[category] = {
                'dates': dates,
                'embeddings': time_embeddings,
                'texts': texts,
                'filename': filename
            }
        
        return results
    
    def create_semantic_network(self, texts: List[str], similarity_threshold: float = 0.7) -> Dict:
        """创建语义网络图"""
        print("🎯 创建语义网络图...")
        
        # 获取嵌入向量
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 计算相似度矩阵
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            for j in range(i+1, n_texts):
                similarity = np.dot(embeddings_array[i], embeddings_array[j]) / (
                    np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i, text in enumerate(texts):
            G.add_node(i, text=text[:50] + '...' if len(text) > 50 else text)
        
        # 添加边（基于相似度阈值）
        for i in range(n_texts):
            for j in range(i+1, n_texts):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # 绘制网络图
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 节点颜色基于聚类（使用K-means）
        from sklearn.cluster import KMeans
        n_clusters = min(4, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        node_labels = kmeans.fit_predict(embeddings_array)
        
        # 绘制节点
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for i in range(n_clusters):
            cluster_nodes = [n for n in G.nodes() if node_labels[n] == i]
            nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, 
                                 node_color=[colors[i]], node_size=1000, alpha=0.8)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edges, width=[w*3 for w in weights], 
                             alpha=0.5, edge_color='gray')
        
        # 绘制标签
        labels = nx.get_node_attributes(G, 'text')
        nx.draw_networkx_labels(G, pos, labels, font_size=8, alpha=0.7)
        
        plt.title('文本语义网络图', fontsize=14)
        plt.axis('off')
        
        filename = "03-advanced/semantic_network.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'graph': G,
            'similarity_matrix': similarity_matrix,
            'node_labels': node_labels,
            'filename': filename
        }
    
    def create_clustering_animation(self, texts: List[str], categories: List[str]) -> str:
        """创建聚类动态展示"""
        print("🎯 创建聚类动态展示...")
        
        # 获取嵌入向量
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(texts)//2))
        embeddings_2d = tsne.fit_transform(embeddings_array)
        
        # 创建动画数据
        frames = []
        
        # K-means聚类过程（多个k值）
        from sklearn.cluster import KMeans
        for k in range(2, min(6, len(texts))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_array)
            
            # 创建帧数据
            frame_data = {
                'k': k,
                'embeddings': embeddings_2d,
                'labels': labels,
                'texts': texts,
                'categories': categories
            }
            frames.append(frame_data)
        
        # 创建交互式动画
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['聚类过程动态展示']
        )
        
        # 为每个k值创建子图
        for i, frame in enumerate(frames):
            df = pd.DataFrame({
                'x': frame['embeddings'][:, 0],
                'y': frame['embeddings'][:, 1],
                'label': frame['labels'],
                'text': frame['texts'],
                'category': frame['categories']
            })
            
            # 添加散点图
            fig.add_trace(
                go.Scatter(
                    x=df['x'], y=df['y'],
                    mode='markers+text',
                    text=df['text'].apply(lambda x: x[:20] + '...'),
                    textposition='top center',
                    textfont_size=8,
                    marker=dict(
                        color=df['label'],
                        colorscale='Viridis',
                        size=10,
                        showscale=True
                    ),
                    name=f'k={frame["k"]}'
                ),
                row=1, col=1
            )
        
        fig.update_layout(
            title='聚类过程动态展示',
            height=600,
            showlegend=True,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': '播放',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000, 'redraw': True}}]
                    }
                ]
            }]
        )
        
        filename = "03-advanced/clustering_animation.html"
        fig.write_html(filename)
        # 在命令行环境不显示，只保存文件
        # fig.show()
        
        return filename
    
    def create_heatmap_visualization(self, texts: List[str], categories: List[str]) -> str:
        """创建相似度热力图"""
        print("🎯 创建相似度热力图...")
        
        # 获取嵌入向量
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 计算相似度矩阵
        n_texts = len(texts)
        similarity_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            for j in range(n_texts):
                similarity = np.dot(embeddings_array[i], embeddings_array[j]) / (
                    np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                )
                similarity_matrix[i][j] = similarity
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        
        # 使用seaborn创建更美观的热力图
        sns.heatmap(similarity_matrix, 
                   xticklabels=[text[:20] + '...' for text in texts],
                   yticklabels=[text[:20] + '...' for text in texts],
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   linewidths=0.5)
        
        plt.title('文本相似度热力图', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = "03-advanced/similarity_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def demo_advanced_visualization(self):
        """演示高级可视化系统"""
        print("🚀 高级功能第3课：高级可视化系统")
        print("=" * 60)
        
        # 加载示例数据
        sample_data = self.load_sample_data()
        
        # 准备数据
        all_texts = []
        all_categories = []
        
        for category, items in sample_data.items():
            for item in items:
                all_texts.append(item['text'])
                all_categories.append(category)
        
        print(f"📊 已加载 {len(all_texts)} 个文本数据")
        
        # 演示1：3D可视化
        print("\n🎯 第1部分：3D空间可视化")
        print("=" * 50)
        result_3d = self.create_3d_visualization(all_texts, all_categories)
        print(f"✅ 3D可视化完成，解释方差比例: {sum(result_3d['explained_variance_ratio']):.3f}")
        
        # 演示2：交互式可视化
        print("\n🎯 第2部分：交互式可视化")
        print("=" * 50)
        interactive_file = self.create_interactive_plotly(all_texts, all_categories)
        print(f"✅ 交互式可视化已保存: {interactive_file}")
        
        # 演示3：时间序列分析
        print("\n🎯 第3部分：时间序列分析")
        print("=" * 50)
        timed_texts = []
        for category, items in sample_data.items():
            for item in items:
                item['category'] = category
                timed_texts.append(item)
        
        time_results = self.create_time_series_analysis(timed_texts)
        print(f"✅ 时间序列分析完成，共分析 {len(time_results)} 个类别")
        
        # 演示4：语义网络图
        print("\n🎯 第4部分：语义网络图")
        print("=" * 50)
        network_result = self.create_semantic_network(all_texts[:8], similarity_threshold=0.6)
        print(f"✅ 语义网络图创建完成，节点数: {len(network_result['graph'].nodes())}, 边数: {len(network_result['graph'].edges())}")
        
        # 演示5：相似度热力图
        print("\n🎯 第5部分：相似度热力图")
        print("=" * 50)
        heatmap_file = self.create_heatmap_visualization(all_texts[:6], all_categories[:6])
        print(f"✅ 相似度热力图已保存: {heatmap_file}")
        
        # 演示6：聚类动画
        print("\n🎯 第6部分：聚类动态展示")
        print("=" * 50)
        animation_file = self.create_clustering_animation(all_texts[:8], all_categories[:8])
        print(f"✅ 聚类动态展示已保存: {animation_file}")
        
        print("\n🎉 高级可视化演示完成！")
        print("\n核心技术总结：")
        print("   • 3D空间可视化")
        print("   • 交互式可视化")
        print("   • 时间序列分析")
        print("   • 语义网络图")
        print("   • 相似度热力图")
        print("   • 聚类动态展示")
        print("\n实际应用场景：")
        print("   • 文本空间可视化工具")
        print("   • 语义关系图")
        print("   • 用户行为可视化")
        print("   • 动态内容分析")
        print("   • 交互式知识探索")
        print("\n下一课：04-performance-optimization.py - 性能优化")

def main():
    """主函数"""
    print("🚀 高级可视化系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        viz_system = AdvancedVisualizationSystem()
        viz_system.demo_advanced_visualization()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
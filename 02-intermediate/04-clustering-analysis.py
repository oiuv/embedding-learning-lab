#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级课程第4课：文本聚类分析
=======================

基于文本嵌入的聚类分析系统实现。
通过向量化技术实现文本数据的无监督聚类和主题发现。

学习目标：
1. 理解聚类分析的工作原理
2. 掌握K-means、层次聚类、DBSCAN算法
3. 实现文本主题发现
4. 聚类结果评估和可视化
5. 处理大规模文本聚类
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

class TextClusteringSystem:
    """文本聚类分析系统"""
    
    def __init__(self):
        """初始化聚类系统"""
        self.client = EmbeddingClient()
        self.embeddings = []
        self.texts = []
        self.labels = []
        
    def load_sample_data(self) -> List[str]:
        """加载示例聚类数据"""
        sample_texts = [
            # 科技类
            "人工智能技术正在改变医疗诊断方式，提高疾病检测准确率",
            "机器学习算法在金融风控中的应用越来越广泛，效果显著",
            "深度学习技术在图像识别领域取得重大突破，准确率超过人类",
            "量子计算机研究取得新进展，有望解决传统计算机无法处理的问题",
            "5G网络技术推动物联网应用快速发展，连接设备数量激增",
            
            # 体育类
            "国足在世界杯预选赛中表现出色，球迷热情高涨期待晋级",
            "NBA总决赛即将打响，湖人队和凯尔特人队争夺总冠军",
            "中国女排在世界锦标赛中获得金牌，展现强大实力赢得尊重",
            "足球世界杯即将开幕，各支球队积极备战争取好成绩",
            "奥运会筹备工作进展顺利，场馆建设完成期待运动员精彩表现",
            
            # 财经类
            "央行宣布降息政策，刺激经济增长应对市场变化",
            "股票市场今日大涨，科技股领涨大盘投资者信心增强",
            "房地产市场调控政策效果显著，房价趋于稳定市场预期改善",
            "国际贸易合作加强，双边贸易额创新高促进经济发展",
            "数字货币试点项目进展顺利，为金融科技创新提供新机遇",
            
            # 娱乐类
            "电影《流浪地球3》票房突破10亿，创影史纪录观众好评如潮",
            "某知名歌手发布新专辑，音乐风格创新大受好评销量领先",
            "电视剧《三体》获得观众一致好评，科幻题材受欢迎讨论热烈",
            "综艺节目创新形式吸引大量年轻观众，收视率和口碑双丰收",
            "某明星慈善活动获得广泛关注，正能量传播社会影响力大"
        ]
        return sample_texts
    
    def prepare_embeddings(self, texts: List[str]) -> np.ndarray:
        """准备文本嵌入"""
        print("🎯 准备文本嵌入...")
        
        embeddings = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        self.embeddings = embeddings_array
        self.texts = texts
        
        print(f"✅ 已处理 {len(texts)} 个文本，嵌入维度: {embeddings_array.shape}")
        return embeddings_array
    
    def kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int = 4) -> Tuple[np.ndarray, Dict]:
        """K-means聚类"""
        print(f"\n🎯 K-means聚类 (k={n_clusters})")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(embeddings, labels)
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        
        results = {
            'algorithm': 'K-means',
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'cluster_centers': cluster_centers,
            'inertia': kmeans.inertia_
        }
        
        return labels, results
    
    def hierarchical_clustering(self, embeddings: np.ndarray, n_clusters: int = 4) -> Tuple[np.ndarray, Dict]:
        """层次聚类"""
        print(f"\n🎯 层次聚类 (k={n_clusters})")
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(embeddings)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(embeddings, labels)
        
        results = {
            'algorithm': 'Hierarchical',
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'linkage': 'ward'
        }
        
        return labels, results
    
    def dbscan_clustering(self, embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> Tuple[np.ndarray, Dict]:
        """DBSCAN聚类"""
        print(f"\n🎯 DBSCAN聚类 (eps={eps}, min_samples={min_samples})")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        # 计算有效聚类的轮廓系数（排除噪声点）
        mask = labels != -1
        if np.sum(mask) > 1:
            silhouette_avg = silhouette_score(embeddings[mask], labels[mask])
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            silhouette_avg = -1
            n_clusters = 0
        
        results = {
            'algorithm': 'DBSCAN',
            'labels': labels,
            'silhouette_score': silhouette_avg,
            'n_clusters': n_clusters,
            'n_noise': np.sum(labels == -1)
        }
        
        return labels, results
    
    def analyze_clusters(self, texts: List[str], labels: np.ndarray, algorithm_name: str) -> Dict:
        """分析聚类结果"""
        print(f"\n📊 {algorithm_name} 聚类结果分析:")
        
        cluster_info = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
                
            cluster_indices = np.where(labels == label)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # 计算聚类大小
            cluster_size = len(cluster_texts)
            
            # 提取关键词（简化版）
            keywords = self.extract_keywords(cluster_texts)
            
            cluster_info[f"聚类_{label}"] = {
                'size': cluster_size,
                'texts': cluster_texts[:3],  # 显示前3个文本
                'keywords': keywords[:5],   # 显示前5个关键词
                'percentage': cluster_size / len(texts) * 100
            }
        
        return cluster_info
    
    def extract_keywords(self, texts: List[str]) -> List[str]:
        """提取关键词（简化版）"""
        # 合并所有文本
        combined_text = ' '.join(texts)
        
        # 简单的关键词提取
        words = combined_text.replace('，', ' ').replace('。', ' ').replace('、', ' ').split()
        word_counts = Counter(words)
        
        # 过滤掉常见词，返回高频词
        common_words = {'的', '了', '在', '是', '和', '与', '为', '对', '中', '上', '下', '这', '那'}
        keywords = [word for word, count in word_counts.most_common() 
                   if word not in common_words and len(word) > 1][:10]
        
        return keywords
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, texts: List[str], 
                          algorithm_name: str):
        """可视化聚类结果"""
        # 设置matplotlib以支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 降维到2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 获取唯一标签
        unique_labels = set(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:  # 噪声点
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          color='gray', marker='x', s=50, label='噪声点')
            else:
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          color=colors[i % len(colors)], alpha=0.7, label=f'聚类{label}')
        
        # 添加文本标签（显示部分文本）
        for i, (x, y) in enumerate(embeddings_2d):
            if labels[i] != -1:  # 不为噪声点
                plt.annotate(f"{i}", (x, y), fontsize=8, alpha=0.7)
        
        plt.title(f"{algorithm_name}聚类结果可视化", fontsize=14)
        plt.xlabel("主成分1", fontsize=12)
        plt.ylabel("主成分2", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        filename = f"02-intermediate/clustering_{algorithm_name.lower()}_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 可视化已保存: {filename}")
    
    def compare_algorithms(self, embeddings: np.ndarray) -> Dict:
        """比较不同聚类算法"""
        print("\n🎯 聚类算法比较")
        print("=" * 50)
        
        algorithms = {
            'K-means': self.kmeans_clustering,
            'Hierarchical': self.hierarchical_clustering,
            'DBSCAN': lambda emb: self.dbscan_clustering(emb, eps=0.6, min_samples=3)
        }
        
        comparison_results = {}
        
        for name, algorithm in algorithms.items():
            try:
                labels, results = algorithm(embeddings)
                comparison_results[name] = results
                
                # 分析聚类结果
                cluster_info = self.analyze_clusters(self.texts, labels, name)
                
                print(f"\n{name}:")
                print(f"   轮廓系数: {results.get('silhouette_score', 'N/A'):.3f}")
                if 'n_clusters' in results:
                    print(f"   聚类数量: {results['n_clusters']}")
                if 'n_noise' in results:
                    print(f"   噪声点数量: {results['n_noise']}")
                
                # 可视化
                self.visualize_clusters(embeddings, labels, self.texts, name)
                
            except Exception as e:
                print(f"   ❌ {name} 执行失败: {e}")
        
        return comparison_results
    
    def find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 8) -> Dict:
        """寻找最优聚类数"""
        print("\n🎯 寻找最优聚类数")
        print("=" * 50)
        
        silhouette_scores = []
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append(score)
            inertias.append(kmeans.inertia_)
            
            print(f"   k={k}: 轮廓系数={score:.3f}, 惯性={kmeans.inertia_:.1f}")
        
        # 找到最优k值
        optimal_k = np.argmax(silhouette_scores) + 2
        
        return {
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'optimal_k': optimal_k,
            'max_silhouette_score': max(silhouette_scores)
        }
    
    def demo_clustering_system(self):
        """演示聚类系统"""
        print("🚀 文本聚类分析演示")
        print("=" * 60)
        
        # 加载示例数据
        texts = self.load_sample_data()
        print(f"📊 已加载 {len(texts)} 个示例文本")
        
        # 准备嵌入
        embeddings = self.prepare_embeddings(texts)
        
        # 寻找最优聚类数
        optimal_info = self.find_optimal_clusters(embeddings)
        print(f"\n📈 最优聚类数: {optimal_info['optimal_k']}")
        print(f"   最高轮廓系数: {optimal_info['max_silhouette_score']:.3f}")
        
        # 使用最优聚类数进行K-means聚类
        print(f"\n🎯 使用最优聚类数 k={optimal_info['optimal_k']} 进行聚类")
        labels, results = self.kmeans_clustering(embeddings, optimal_info['optimal_k'])
        
        # 分析聚类结果
        cluster_info = self.analyze_clusters(texts, labels, "最优K-means")
        
        print("\n📊 聚类结果:")
        for cluster_name, info in cluster_info.items():
            print(f"\n{cluster_name}:")
            print(f"   大小: {info['size']} ({info['percentage']:.1f}%)")
            print(f"   关键词: {', '.join(info['keywords'])}")
            print(f"   示例文本: {info['texts'][0][:50]}...")
        
        # 比较不同算法
        print("\n🎯 第5部分：算法比较")
        comparison_results = self.compare_algorithms(embeddings)
        
        print("\n🎉 聚类分析演示完成！")
        print("\n核心技术总结:")
        print("   • K-means聚类")
        print("   • 层次聚类")
        print("   • DBSCAN聚类")
        print("   • 轮廓系数评估")
        print("   • 聚类可视化")
        print("\n实际应用场景:")
        print("   • 文档主题发现")
        print("   • 新闻聚类")
        print("   • 客户反馈分析")
        print("   • 社交媒体内容分组")
        print("   • 推荐系统用户分群")

def main():
    """主函数"""
    print("🚀 中级课程第4课：文本聚类分析")
    print("=" * 60)
    print("基于文本嵌入的聚类分析系统实现。\n")
    
    try:
        # 检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("🔑 API密钥检查")
            print("-" * 30)
            print("⚠️ 未检测到 DASHSCOPE_API_KEY 环境变量")
            print("\n解决方法：")
            print("1. 临时设置: set DASHSCOPE_API_KEY=你的密钥 (Windows)")
            print("2. 临时设置: export DASHSCOPE_API_KEY=你的密钥 (Linux/Mac)")
            print("\n📝 获取API密钥：")
            print("   访问 https://dashscope.console.aliyun.com 申请")
            return
        else:
            print("✅ 检测到API密钥")
        
        input("\n📊 按回车键开始聚类分析演示...")
        print("\n" + "="*60)
        clustering_system = TextClusteringSystem()
        clustering_system.demo_clustering_system()
        
        print("\n" + "="*60)
        print("🎉 聚类分析课程完成！")
        print("🎯 你已经掌握了：")
        print("✅ K-means聚类")
        print("✅ 层次聚类")
        print("✅ DBSCAN聚类")
        print("✅ 轮廓系数评估")
        print("✅ 聚类可视化")
        print("\n📂 可视化结果已保存为 clustering_*.png")
        print("\n🚀 实际应用场景:")
        print("   • 文档主题发现")
        print("   • 新闻聚类")
        print("   • 客户反馈分析")
        print("   • 社交媒体内容分组")
        print("   • 推荐系统用户分群")
        print("\n🎓 恭喜你完成了中级课程！")
        print("\n🎯 准备进入高级模块...")
        print("\n高级模块：03-advanced/01-knowledge-base.py - 智能知识库")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("🔄 请检查网络连接和API配置")
    finally:
        input("\n📚 按回车键退出课程...")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4课：向量操作基础
============================

本课程将教你如何处理和操作文本嵌入向量。

学习目标：
1. 理解向量的基本操作
2. 掌握向量加减乘除
3. 学习向量归一化
4. 实现向量降维
5. 理解向量空间中的语义运算

"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class VectorOperations:
    """向量操作类"""
    
    def __init__(self, api_key: str = None):
        """初始化"""
        try:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "text-embedding-v4"
            self.dimensions = 1024
            print("✅ 向量操作器初始化成功！")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 获取嵌入失败: {e}")
            return []
    
    def vector_addition(self, vec1: List[float], vec2: List[float]) -> List[float]:
        """向量加法"""
        return (np.array(vec1) + np.array(vec2)).tolist()
    
    def vector_subtraction(self, vec1: List[float], vec2: List[float]) -> List[float]:
        """向量减法"""
        return (np.array(vec1) - np.array(vec2)).tolist()
    
    def vector_scaling(self, vec: List[float], scale: float) -> List[float]:
        """向量缩放"""
        return (np.array(vec) * scale).tolist()
    
    def vector_norm(self, vec: List[float]) -> float:
        """计算向量范数"""
        return np.linalg.norm(vec)
    
    def vector_normalize(self, vec: List[float]) -> List[float]:
        """向量归一化"""
        norm = self.vector_norm(vec)
        if norm == 0:
            return vec
        return (np.array(vec) / norm).tolist()
    
    def vector_dot_product(self, vec1: List[float], vec2: List[float]) -> float:
        """计算点积"""
        return np.dot(vec1, vec2)
    
    def demonstrate_basic_operations(self):
        """演示基本向量操作"""
        print("🎯 第1部分：基本向量操作")
        print("=" * 50)
        
        # 获取两个文本的向量
        text1 = "国王"
        text2 = "男人"
        text3 = "女人"
        
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        vec3 = self.get_embedding(text3)
        
        if not all([vec1, vec2, vec3]):
            print("❌ 获取嵌入失败")
            return
        
        # 演示各种操作
        print(f"📊 原始向量范数:")
        print(f"   '{text1}': {self.vector_norm(vec1):.4f}")
        print(f"   '{text2}': {self.vector_norm(vec2):.4f}")
        print(f"   '{text3}': {self.vector_norm(vec3):.4f}")
        
        # 向量加法
        addition = self.vector_addition(vec1, vec2)
        print(f"\n➕ 向量加法 '{text1}' + '{text2}':")
        print(f"   结果范数: {self.vector_norm(addition):.4f}")
        
        # 向量减法
        subtraction = self.vector_subtraction(vec1, vec2)
        print(f"\n➖ 向量减法 '{text1}' - '{text2}':")
        print(f"   结果范数: {self.vector_norm(subtraction):.4f}")
        
        # 向量缩放
        scaled = self.vector_scaling(vec1, 2.0)
        print(f"\n📏 向量缩放 '{text1}' × 2:")
        print(f"   原始范数: {self.vector_norm(vec1):.4f}")
        print(f"   缩放后范数: {self.vector_norm(scaled):.4f}")
        
        # 向量归一化
        normalized = self.vector_normalize(vec1)
        print(f"\n🎯 向量归一化 '{text1}':")
        print(f"   归一化后范数: {self.vector_norm(normalized):.4f}")
        
        # 点积
        dot_product = self.vector_dot_product(vec1, vec2)
        print(f"\n🔢 点积 '{text1}' · '{text2}': {dot_product:.4f}")
    
    def demonstrate_semantic_analogy(self):
        """演示语义类比"""
        print("\n🎯 第2部分：语义类比运算")
        print("=" * 50)
        
        # 经典的国王-男人+女人=女王类比
        texts = ["国王", "男人", "女人", "女王"]
        embeddings = {}
        
        for text in texts:
            embeddings[text] = self.get_embedding(text)
        
        # 执行类比运算
        king = np.array(embeddings["国王"])
        man = np.array(embeddings["男人"])
        woman = np.array(embeddings["女人"])
        queen = np.array(embeddings["女王"])
        
        # 计算国王 - 男人 + 女人
        analogy_result = king - man + woman
        
        # 计算与女王的相似度
        similarity = np.dot(analogy_result, queen) / (np.linalg.norm(analogy_result) * np.linalg.norm(queen))
        
        print("📝 语义类比运算:")
        print(f"   公式: 国王 - 男人 + 女人 ≈ 女王")
        print(f"   计算结果与'女王'的相似度: {similarity:.4f}")
        
        # 验证其他可能的答案
        candidates = ["女王", "公主", "皇后", "王后", "皇帝"]
        candidate_embeddings = {cand: self.get_embedding(cand) for cand in candidates}
        
        print("\n🔍 验证其他候选词:")
        for cand, emb in candidate_embeddings.items():
            cand_vec = np.array(emb)
            sim = np.dot(analogy_result, cand_vec) / (np.linalg.norm(analogy_result) * np.linalg.norm(cand_vec))
            print(f"   {cand}: {sim:.4f}")
    
    def demonstrate_vector_clustering(self):
        """演示向量聚类"""
        print("\n🎯 第3部分：向量聚类分析")
        print("=" * 50)
        
        # 准备不同类别的文本
        categories = {
            "技术": ["人工智能", "机器学习", "深度学习", "神经网络"],
            "美食": ["披萨", "汉堡", "寿司", "火锅"],
            "运动": ["足球", "篮球", "游泳", "跑步"]
        }
        
        # 获取所有文本的嵌入
        all_texts = []
        all_embeddings = []
        labels = []
        
        for category, texts in categories.items():
            for text in texts:
                all_texts.append(text)
                labels.append(category)
                all_embeddings.append(self.get_embedding(text))
        
        # 转换为numpy数组
        embeddings_array = np.array(all_embeddings)
        
        # 计算类别中心
        category_centers = {}
        for category, texts in categories.items():
            indices = [i for i, label in enumerate(labels) if label == category]
            category_embeddings = embeddings_array[indices]
            category_centers[category] = np.mean(category_embeddings, axis=0)
        
        print("📊 类别中心距离分析:")
        for cat1, center1 in category_centers.items():
            for cat2, center2 in category_centers.items():
                if cat1 != cat2:
                    distance = np.linalg.norm(center1 - center2)
                    print(f"   {cat1} 与 {cat2} 的距离: {distance:.2f}")
        
        return all_texts, all_embeddings, labels
    
    def demonstrate_dimensionality_reduction(self):
        """演示降维技术"""
        print("\n🎯 第4部分：降维可视化")
        print("=" * 50)
        
        # 获取示例文本
        texts = [
            "人工智能", "机器学习", "深度学习", "神经网络",
            "披萨", "汉堡", "寿司", "火锅",
            "足球", "篮球", "游泳", "跑步",
            "汽车", "飞机", "火车", "自行车"
        ]
        
        # 获取高维嵌入
        embeddings = [self.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        print(f"📊 原始维度: {embeddings_array.shape}")
        
        # PCA降维到2D
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(embeddings_array)
        
        # t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        embeddings_2d_tsne = tsne.fit_transform(embeddings_array)
        
        # 设置matplotlib以支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # PCA可视化
        ax1.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], alpha=0.6)
        for i, text in enumerate(texts):
            ax1.annotate(text, (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]), 
                        fontsize=9, alpha=0.8)
        ax1.set_title("PCA降维到2D")
        ax1.grid(True, alpha=0.3)
        
        # t-SNE可视化
        ax2.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], alpha=0.6)
        for i, text in enumerate(texts):
            ax2.annotate(text, (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]), 
                        fontsize=9, alpha=0.8)
        ax2.set_title("t-SNE降维到2D")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('01-basics/vector_operations_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 降维可视化已保存")
        
        # 计算降维后的信息保留
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"📈 PCA保留的方差比例: {explained_variance:.2%}")
    
    def demonstrate_vector_statistics(self):
        """演示向量统计分析"""
        print("\n🎯 第5部分：向量统计分析")
        print("=" * 50)
        
        # 获取多个文本的嵌入
        texts = ["人工智能", "机器学习", "深度学习", "神经网络"]
        embeddings = [self.get_embedding(text) for text in texts]
        
        # 转换为numpy数组
        embeddings_array = np.array(embeddings)
        
        # 计算统计信息
        print("📊 嵌入统计信息:")
        print(f"   文本数量: {len(texts)}")
        print(f"   向量维度: {embeddings_array.shape[1]}")
        print(f"   向量均值: {np.mean(embeddings_array):.4f}")
        print(f"   向量标准差: {np.std(embeddings_array):.4f}")
        
        # 计算每维度的统计
        dim_means = np.mean(embeddings_array, axis=0)
        dim_stds = np.std(embeddings_array, axis=0)
        
        print(f"\n📈 维度统计 (前10维):")
        for i in range(min(10, len(dim_means))):
            print(f"   维度 {i}: 均值={dim_means[i]:.4f}, 标准差={dim_stds[i]:.4f}")
        
        # 计算文本间的距离矩阵
        distance_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(len(texts)):
                distance_matrix[i][j] = np.linalg.norm(embeddings_array[i] - embeddings_array[j])
        
        print(f"\n📏 距离矩阵:")
        print("文本:", texts)
        print(distance_matrix)
        
        return embeddings_array

def main():
    """主函数"""
    print("🚀 第4课：向量操作基础")
    print("=" * 60)
    print("本课程将教你如何操作和分析文本嵌入向量。\n")
    
    try:
        # 创建操作器实例
        operator = VectorOperations()
        
        input("📐 按回车键开始学习基本向量操作...")
        print("\n" + "="*60)
        operator.demonstrate_basic_operations()
        
        input("\n🧠 按回车键体验语义类比运算...")
        print("\n" + "="*60)
        operator.demonstrate_semantic_analogy()
        
        input("\n📊 按回车键进行向量聚类分析...")
        print("\n" + "="*60)
        texts, embeddings, labels = operator.demonstrate_vector_clustering()
        
        input("\n📉 按回车键学习降维可视化...")
        print("\n" + "="*60)
        operator.demonstrate_dimensionality_reduction()
        
        input("\n📈 按回车键进行向量统计分析...")
        print("\n" + "="*60)
        embeddings_array = operator.demonstrate_vector_statistics()
        
        print("\n" + "="*60)
        print("🎉 第4课完成！")
        print("🎓 基础课程已全部完成！")
        print("\n你已经掌握了：")
        print("✅ 基本向量操作（加减乘除）")
        print("✅ 语义类比运算")
        print("✅ 向量聚类分析")
        print("✅ 降维可视化技术")
        print("✅ 向量统计分析")
        print("\n📂 可视化结果已保存为 'vector_operations_visualization.png'")
        print("\n🚀 恭喜你完成了基础课程！")
        print("\n🎯 准备进入中级应用阶段...")
        print("\n中级模块：02-intermediate/01-semantic-search.py - 语义搜索")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {str(e)}")
        print("🔄 请检查网络连接和API配置后重试")
    finally:
        input("\n📚 按回车键退出基础课程...")

if __name__ == "__main__":
    main()
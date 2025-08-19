#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础练习1：相似度计算挑战
====================

完成多种文本相似度计算方法的实现和比较。

练习目标：
1. 实现余弦相似度、欧几里得距离、曼哈顿距离
2. 比较不同方法的计算结果
3. 理解相似度阈值的作用
4. 可视化相似度矩阵
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

class SimilarityChallenge:
    """相似度计算挑战"""
    
    def __init__(self):
        self.client = EmbeddingClient()
    
    def load_challenge_data(self) -> List[str]:
        """加载挑战数据"""
        texts = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的重要技术",
            "深度学习是机器学习的高级形式",
            "Python是流行的编程语言",
            "JavaScript用于Web开发",
            "数据科学结合了统计和编程",
            "自然语言处理让计算机理解人类语言",
            "计算机视觉让计算机看懂图像"
        ]
        return texts
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度
        
        挑战：实现余弦相似度计算
        公式：cos(θ) = (A·B) / (||A|| * ||B||)
        """
        # TODO: 实现余弦相似度计算
        # 提示：使用np.dot计算点积，np.linalg.norm计算范数
        
        # 你的代码开始
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        # 你的代码结束
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算欧几里得距离
        
        挑战：实现欧几里得距离计算
        公式：d = √(Σ(ai - bi)²)
        """
        # TODO: 实现欧几里得距离计算
        # 提示：使用np.sqrt和np.sum
        
        # 你的代码开始
        diff = vec1 - vec2
        return np.sqrt(np.sum(diff ** 2))
        # 你的代码结束
    
    def manhattan_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算曼哈顿距离
        
        挑战：实现曼哈顿距离计算
        公式：d = Σ|ai - bi|
        """
        # TODO: 实现曼哈顿距离计算
        # 提示：使用np.abs和np.sum
        
        # 你的代码开始
        return np.sum(np.abs(vec1 - vec2))
        # 你的代码结束
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """计算Jaccard相似度
        
        挑战：实现Jaccard相似度计算
        公式：|A∩B| / |A∪B|
        """
        # TODO: 实现Jaccard相似度计算
        # 提示：使用集合操作
        
        # 你的代码开始
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
        # 你的代码结束
    
    def build_similarity_matrix(self, embeddings: np.ndarray, method: str = 'cosine') -> np.ndarray:
        """构建相似度矩阵
        
        挑战：为给定的嵌入向量构建相似度矩阵
        """
        n = len(embeddings)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if method == 'cosine':
                    matrix[i][j] = self.cosine_similarity(embeddings[i], embeddings[j])
                elif method == 'euclidean':
                    # 将距离转换为相似度
                    distance = self.euclidean_distance(embeddings[i], embeddings[j])
                    matrix[i][j] = 1 / (1 + distance)
                elif method == 'manhattan':
                    # 将距离转换为相似度
                    distance = self.manhattan_distance(embeddings[i], embeddings[j])
                    matrix[i][j] = 1 / (1 + distance)
        
        return matrix
    
    def find_similar_pairs(self, texts: List[str], threshold: float = 0.7) -> List[Tuple[int, int, float]]:
        """查找相似文本对
        
        挑战：找出所有相似度超过阈值的文本对
        """
        # 获取嵌入
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 计算相似度矩阵
        similarity_matrix = self.build_similarity_matrix(embeddings_array)
        
        # 找出相似对
        similar_pairs = []
        n = len(texts)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    similar_pairs.append((i, j, similarity))
        
        return similar_pairs
    
    def visualize_similarity_matrix(self, texts: List[str], method: str = 'cosine'):
        """可视化相似度矩阵"""
        # 获取嵌入
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 构建相似度矩阵
        matrix = self.build_similarity_matrix(embeddings_array, method)
        
        # 创建可视化
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(label='相似度')
        plt.title(f'文本相似度矩阵 ({method})')
        plt.xlabel('文档索引')
        plt.ylabel('文档索引')
        
        # 添加文本标签
        for i in range(len(texts)):
            for j in range(len(texts)):
                plt.text(j, i, f'{matrix[i][j]:.2f}', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('05-exercises/basic/similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_challenge(self):
        """运行挑战"""
        print("🎯 基础练习1：相似度计算挑战")
        print("=" * 50)
        
        # 加载数据
        texts = self.load_challenge_data()
        print(f"📊 加载 {len(texts)} 个文本")
        
        # 获取嵌入
        print("🔄 获取文本嵌入...")
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 测试相似度计算方法
        print("\n🧮 测试相似度计算方法")
        print("-" * 30)
        
        # 测试余弦相似度
        cos_sim = self.cosine_similarity(embeddings_array[0], embeddings_array[1])
        print(f"余弦相似度(文本0, 文本1): {cos_sim:.4f}")
        
        # 测试欧几里得距离
        euclid_dist = self.euclidean_distance(embeddings_array[0], embeddings_array[1])
        print(f"欧几里得距离(文本0, 文本1): {euclid_dist:.4f}")
        
        # 测试曼哈顿距离
        manhattan_dist = self.manhattan_distance(embeddings_array[0], embeddings_array[1])
        print(f"曼哈顿距离(文本0, 文本1): {manhattan_dist:.4f}")
        
        # 测试Jaccard相似度
        words1 = set(texts[0].split())
        words2 = set(texts[1].split())
        jaccard = self.jaccard_similarity(words1, words2)
        print(f"Jaccard相似度(文本0, 文本1): {jaccard:.4f}")
        
        # 查找相似对
        similar_pairs = self.find_similar_pairs(texts, threshold=0.7)
        print(f"\n🔗 相似度≥0.7的文本对: {len(similar_pairs)}")
        for i, j, sim in similar_pairs[:3]:
            print(f"   文本{i} - 文本{j}: {sim:.3f}")
        
        # 可视化
        print("\n📊 生成可视化...")
        self.visualize_similarity_matrix(texts, method='cosine')
        
        print("\n✅ 挑战完成！")
        print("\n🎓 学习要点：")
        print("   • 余弦相似度：衡量向量方向相似性")
        print("   • 欧几里得距离：衡量向量空间距离")
        print("   • 曼哈顿距离：衡量坐标轴距离和")
        print("   • Jaccard相似度：衡量集合重叠度")

def main():
    """主函数"""
    print("🚀 基础练习：相似度计算挑战")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        challenge = SimilarityChallenge()
        challenge.run_challenge()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
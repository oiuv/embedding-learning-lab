#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3课：计算文本相似度
========================

本课程将教你如何计算文本之间的相似度。

学习目标：
1. 理解相似度计算方法
2. 实现余弦相似度计算
3. 比较不同文本的相似性
4. 理解相似度阈值的意义

"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
from openai import OpenAI

class SimilarityCalculator:
    """文本相似度计算器"""
    
    def __init__(self, api_key: str = None):
        """初始化计算器"""
        try:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "text-embedding-v4"
            self.dimensions = 1024
            print("✅ 相似度计算器初始化成功！")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
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
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # 计算点积
        dot_product = np.dot(v1, v2)
        
        # 计算范数
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # 避免除以零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))  # 确保范围在[-1, 1]
    
    def euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """计算欧氏距离"""
        return np.linalg.norm(np.array(vec1) - np.array(vec2))
    
    def manhattan_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """计算曼哈顿距离"""
        return np.sum(np.abs(np.array(vec1) - np.array(vec2)))
    
    def demonstrate_similarity_calculation(self):
        """演示相似度计算"""
        print("🎯 第1部分：基础相似度计算")
        print("=" * 50)
        
        # 示例文本
        text1 = "机器学习"
        text2 = "深度学习"
        text3 = "苹果"
        
        # 获取嵌入
        print("📊 获取文本嵌入...")
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        emb3 = self.get_embedding(text3)
        
        if not all([emb1, emb2, emb3]):
            print("❌ 获取嵌入失败")
            return
        
        # 计算相似度
        sim_12 = self.cosine_similarity(emb1, emb2)
        sim_13 = self.cosine_similarity(emb1, emb3)
        sim_23 = self.cosine_similarity(emb2, emb3)
        
        print(f"\n📈 相似度结果:")
        print(f"   '{text1}' vs '{text2}': {sim_12:.4f}")
        print(f"   '{text1}' vs '{text3}': {sim_13:.4f}")
        print(f"   '{text2}' vs '{text3}': {sim_23:.4f}")
        
        # 解释结果
        print(f"\n📝 结果分析:")
        print(f"   • 余弦相似度范围: [-1, 1]")
        print(f"   • 1.0: 完全相同")
        print(f"   • 0.0: 完全不相关")
        print(f"   • -1.0: 完全相反")
    
    def compare_different_similarity_methods(self):
        """比较不同的相似度计算方法"""
        print("\n🎯 第2部分：不同相似度方法比较")
        print("=" * 50)
        
        # 示例文本对
        text_pairs = [
            ("苹果", "香蕉"),
            ("机器学习", "深度学习"),
            ("汽车", "飞机"),
            ("高兴", "快乐")
        ]
        
        # 获取所有文本的嵌入
        all_texts = list(set([text for pair in text_pairs for text in pair]))
        embeddings = {}
        
        for text in all_texts:
            embeddings[text] = self.get_embedding(text)
        
        # 计算不同方法的相似度
        print("\n📊 相似度比较:")
        print(f"{'文本对':<20} {'余弦':<10} {'欧氏距离':<10} {'曼哈顿':<10}")
        print("-" * 50)
        
        for text1, text2 in text_pairs:
            emb1 = embeddings[text1]
            emb2 = embeddings[text2]
            
            cosine = self.cosine_similarity(emb1, emb2)
            euclidean = self.euclidean_distance(emb1, emb2)
            manhattan = self.manhattan_distance(emb1, emb2)
            
            print(f"{text1} vs {text2:<12} {cosine:.4f}   {euclidean:.2f}     {manhattan:.2f}")
    
    def find_most_similar_texts(self, query: str, candidates: List[str], top_k: int = 3):
        """查找最相似的文本"""
        print("\n🎯 第3部分：查找最相似文本")
        print("=" * 50)
        
        # 获取查询嵌入
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # 批量获取候选文本嵌入
        candidate_embeddings = self.get_embeddings_batch(candidates)
        
        # 计算相似度
        similarities = []
        for candidate, embedding in zip(candidates, candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, embedding)
            similarities.append((candidate, sim))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]
        
        print(f"🔍 查询: '{query}'")
        print(f"📊 最相似的{top_k}个文本:")
        
        for i, (text, sim) in enumerate(top_similar, 1):
            print(f"   {i}. '{text}' (相似度: {sim:.4f})")
        
        return top_similar
    
    def demonstrate_similarity_threshold(self):
        """演示相似度阈值的应用"""
        print("\n🎯 第4部分：相似度阈值应用")
        print("=" * 50)
        
        # 示例文本
        documents = [
            "人工智能技术",
            "机器学习算法",
            "深度学习框架",
            "苹果公司的iPhone",
            "香蕉的营养价值",
            "深度学习需要大量数据",
            "机器学习是AI的子集"
        ]
        
        query = "人工智能"
        
        # 获取嵌入
        query_embedding = self.get_embedding(query)
        doc_embeddings = self.get_embeddings_batch(documents)
        
        # 不同阈值下的结果
        thresholds = [0.5, 0.7, 0.8, 0.9]
        
        for threshold in thresholds:
            similar_docs = []
            for doc, emb in zip(documents, doc_embeddings):
                sim = self.cosine_similarity(query_embedding, emb)
                if sim >= threshold:
                    similar_docs.append((doc, sim))
            
            similar_docs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 阈值 {threshold} 的结果:")
            print(f"   匹配数量: {len(similar_docs)}")
            for doc, sim in similar_docs:
                print(f"   - '{doc}' ({sim:.3f})")
    
    def create_similarity_matrix(self, texts: List[str]):
        """创建相似度矩阵"""
        print("\n🎯 第5部分：相似度矩阵")
        print("=" * 50)
        
        # 获取所有文本的嵌入
        embeddings = self.get_embeddings_batch(texts)
        
        # 计算相似度矩阵
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i][j] = self.cosine_similarity(embeddings[i], embeddings[j])
        
        # 打印矩阵
        print("📊 相似度矩阵:")
        print("文本:", texts)
        print("\n矩阵:")
        print(similarity_matrix)
        
        # 保存矩阵到文件
        np.savetxt('01-basics/similarity_matrix.txt', similarity_matrix, fmt='%.4f')
        print("\n✅ 相似度矩阵已保存到 'similarity_matrix.txt'")
        
        return similarity_matrix

def main():
    """主函数"""
    print("🚀 第3课：计算文本相似度")
    print("=" * 60)
    print("本课程将教你如何计算文本之间的相似度。\n")
    
    try:
        # 创建计算器实例
        calculator = SimilarityCalculator()
        
        input("📊 按回车键开始基础相似度计算...")
        print("\n" + "="*60)
        calculator.demonstrate_similarity_calculation()
        
        input("\n🔍 按回车键比较不同相似度方法...")
        print("\n" + "="*60)
        calculator.compare_different_similarity_methods()
        
        # 查找相似文本
        sample_texts = [
            "人工智能技术",
            "机器学习算法",
            "深度学习框架",
            "自然语言处理",
            "计算机视觉",
            "数据科学",
            "神经网络",
            "Python编程",
            "Java开发",
            "Web开发"
        ]
        
        input(f"\n🔎 按回车键查找最相似文本 (查询: '人工智能')...")
        print("\n" + "="*60)
        calculator.find_most_similar_texts("人工智能", sample_texts, top_k=5)
        
        input("\n📏 按回车键学习相似度阈值应用...")
        print("\n" + "="*60)
        calculator.demonstrate_similarity_threshold()
        
        input("\n📈 按回车键创建相似度矩阵...")
        print("\n" + "="*60)
        calculator.create_similarity_matrix(sample_texts[:5])
        
        print("\n" + "="*60)
        print("🎉 第3课完成！")
        print("你已经学会了：")
        print("✅ 余弦相似度计算")
        print("✅ 不同相似度方法比较")
        print("✅ 查找最相似文本")
        print("✅ 使用相似度阈值")
        print("✅ 创建相似度矩阵")
        print("\n📂 相似度矩阵已保存到 'similarity_matrix.txt'")
        print("\n🎯 下一课：04-vector-operations.py - 向量操作基础")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {str(e)}")
        print("🔄 请检查网络连接和API配置后重试")
    finally:
        input("\n📚 按回车键退出课程...")

if __name__ == "__main__":
    main()
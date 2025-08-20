#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级课程第3课：文本推荐系统
=======================

基于官方示例的文本推荐系统实现。
通过文本嵌入实现文章标题的个性化推荐。

学习目标：
1. 理解推荐系统的工作原理
2. 掌握缓存机制的使用
3. 实现基于内容的推荐
4. 优化推荐算法性能
5. 处理大规模文本数据
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

class TextRecommendationSystem:
    """文本推荐系统"""
    
    def __init__(self, cache_file: str = "text_recommendations.pkl"):
        """初始化推荐系统"""
        self.client = EmbeddingClient()
        self.cache_file = cache_file
        self.embedding_cache = {}
        self.texts = []
        
    def load_sample_data(self) -> List[str]:
        """加载示例数据（模拟新闻标题）"""
        sample_titles = [
            "人工智能技术在医疗领域的突破性应用",
            "机器学习算法优化提升搜索引擎准确性",
            "深度学习在图像识别中的最新进展",
            "Python数据科学库的新版本发布",
            "自然语言处理技术改变客户服务体验",
            "区块链技术在金融行业的创新应用",
            "云计算服务降低企业IT成本",
            "大数据分析方法助力精准营销",
            "物联网技术推动智能家居发展",
            "量子计算研究取得重要突破",
            "5G网络技术改变通信行业格局",
            "虚拟现实技术在教育领域的应用",
            "自动驾驶技术面临的安全挑战",
            "新能源技术推动可持续发展",
            "网络安全技术应对新型威胁"
        ]
        return sample_titles
    
    def get_embedding_with_cache(self, text: str) -> List[float]:
        """获取嵌入向量并缓存"""
        if text not in self.embedding_cache:
            embedding = self.client.get_embedding(text)
            if embedding:
                self.embedding_cache[text] = embedding
        return self.embedding_cache.get(text, [])
    
    def save_cache(self):
        """保存嵌入缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"✅ 缓存已保存：{len(self.embedding_cache)}个嵌入")
        except Exception as e:
            print(f"❌ 保存缓存失败：{e}")
    
    def load_cache(self) -> bool:
        """加载嵌入缓存"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            print(f"✅ 缓存已加载：{len(self.embedding_cache)}个嵌入")
            return True
        except FileNotFoundError:
            print("🆕 未找到缓存文件")
            return False
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def setup_system(self, texts: List[str]) -> bool:
        """设置推荐系统"""
        self.texts = texts
        
        # 尝试加载缓存
        cache_loaded = self.load_cache()
        
        # 检查需要处理的文本
        texts_to_process = [t for t in texts if t not in self.embedding_cache]
        
        if texts_to_process:
            print(f"🔄 正在处理{len(texts_to_process)}个新文本...")
            embeddings = self.client.get_embeddings_batch(texts_to_process)
            
            for text, embedding in zip(texts_to_process, embeddings):
                if embedding:
                    self.embedding_cache[text] = embedding
            
            self.save_cache()
        
        return True
    
    def get_recommendations(self, query_text: str, k: int = 3) -> List[Dict]:
        """获取文本推荐"""
        if not self.texts:
            return []
        
        # 获取查询文本的嵌入
        query_embedding = self.get_embedding_with_cache(query_text)
        if not query_embedding:
            return []
        
        # 计算相似度
        similarities = []
        for text in self.texts:
            text_embedding = self.get_embedding_with_cache(text)
            if text_embedding:
                similarity = self.cosine_similarity(query_embedding, text_embedding)
                similarities.append((text, similarity))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:k]
        
        return [
            {"text": text, "similarity": score}
            for text, score in top_results
        ]
    
    def recommend_similar_articles(self, index: int, k: int = 3) -> List[Dict]:
        """推荐相似文章"""
        if index < 0 or index >= len(self.texts):
            return []
        
        source_text = self.texts[index]
        source_embedding = self.get_embedding_with_cache(source_text)
        
        if not source_embedding:
            return []
        
        similarities = []
        for i, text in enumerate(self.texts):
            if i == index:
                continue
            
            text_embedding = self.get_embedding_with_cache(text)
            if text_embedding:
                similarity = self.cosine_similarity(source_embedding, text_embedding)
                similarities.append((text, similarity, i))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:k]
        
        return [
            {"text": text, "similarity": score, "index": idx}
            for text, score, idx in top_results
        ]
    
    def get_similarity_matrix(self) -> np.ndarray:
        """获取相似度矩阵"""
        if not self.texts:
            return np.array([])
        
        n = len(self.texts)
        matrix = np.zeros((n, n))
        
        embeddings = [self.get_embedding_with_cache(text) for text in self.texts]
        
        for i in range(n):
            for j in range(n):
                if embeddings[i] and embeddings[j]:
                    matrix[i][j] = self.cosine_similarity(embeddings[i], embeddings[j])
        
        return matrix
    
    def demo_recommendation_system(self):
        """演示推荐系统"""
        print("🎯 文本推荐系统演示")
        print("=" * 40)
        
        # 加载示例数据
        texts = self.load_sample_data()
        print(f"📊 已加载{len(texts)}个文本")
        
        # 设置系统
        self.setup_system(texts)
        
        # 演示推荐功能
        print("\n🔍 基于文本的推荐演示：")
        
        test_queries = [
            "人工智能的发展趋势",
            "编程语言和框架",
            "网络安全和数据保护",
            "新能源技术应用"
        ]
        
        for query in test_queries:
            print(f"\n📝 查询：'{query}'")
            recommendations = self.get_recommendations(query, k=2)
            
            for rec in recommendations:
                print(f"   📄 {rec['text'][:50]}... (相似度：{rec['similarity']:.3f})")
        
        # 演示相似文章推荐
        print("\n🔗 相似文章推荐演示：")
        
        for idx in [0, 5, 10]:  # 选择几个不同的文章
            if idx < len(texts):
                print(f"\n基于第{idx+1}篇文章：{texts[idx][:30]}...")
                similar = self.recommend_similar_articles(idx, k=2)
                
                for rec in similar:
                    print(f"   📄 {rec['text'][:50]}... (相似度：{rec['similarity']:.3f})")

def main():
    """主函数"""
    print("🚀 中级课程第3课：文本推荐系统")
    print("=" * 60)
    print("通过文本嵌入实现文章标题的个性化推荐。\n")
    
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
        
        input("\n📰 按回车键开始推荐系统演示...")
        print("\n" + "="*60)
        system = TextRecommendationSystem()
        system.demo_recommendation_system()
        
        print("\n" + "="*60)
        print("🎉 推荐系统课程完成！")
        print("🎯 你已经掌握了：")
        print("✅ 基于内容的推荐")
        print("✅ 相似度计算")
        print("✅ 缓存机制优化")
        print("✅ 大规模数据处理")
        print("\n🚀 实际应用场景:")
        print("   • 文章推荐系统")
        print("   • 产品推荐引擎")
        print("   • 个性化内容推荐")
        print("   • 用户兴趣建模")
        print("\n📂 缓存文件已保存为 'text_recommendations.pkl'")
        print("\n🎯 准备进入下一课程...")
        print("\n中级模块：04-clustering-analysis.py - 聚类分析")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行错误：{e}")
        print("🔄 请检查网络连接和API配置")
    finally:
        input("\n📚 按回车键退出课程...")

if __name__ == "__main__":
    main()
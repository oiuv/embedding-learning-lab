#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级课程第2课：文本分类系统 - 产品评论示例
=========================================

通过文本嵌入技术实现产品评论的自动分类和情感分析。
展示如何将文本分类技术应用于电商场景。

学习目标：
1. 理解文本分类在实际业务中的应用
2. 掌握评论数据的向量化处理
3. 实现多类别自动分类
4. 添加缓存机制提升性能
5. 处理大规模评论数据
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

class ProductReviewSearchSystem:
    """产品评论语义搜索系统"""
    
    def __init__(self, api_key: str = None, cache_file: str = "review_embeddings.pkl"):
        """初始化系统"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v4"
        self.dimensions = 1024
        self.cache_file = cache_file
        self.reviews_df = None
        self.embedding_cache = {}
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        # 分批处理，避免API限制
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                all_embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"❌ 生成嵌入失败: {e}")
                return []
        
        return all_embeddings
    
    def load_sample_data(self) -> pd.DataFrame:
        """加载示例数据（模拟美食评论数据）"""
        sample_data = [
            {
                "product_name": "有机蜂蜜",
                "review_text": "这款蜂蜜口感醇厚，甜度适中，包装精美，很适合送礼。",
                "score": 5.0,
                "category": "食品"
            },
            {
                "product_name": "进口咖啡豆",
                "review_text": "咖啡豆新鲜，香气浓郁，冲泡后口感顺滑，是我喝过的最好的咖啡之一。",
                "score": 4.8,
                "category": "饮料"
            },
            {
                "product_name": "有机燕麦",
                "review_text": "燕麦片很新鲜，口感很好，搭配牛奶或酸奶都很美味，早餐必备！",
                "score": 4.5,
                "category": "食品"
            },
            {
                "product_name": "特级橄榄油",
                "review_text": "橄榄油品质很好，味道纯正，适合凉拌和低温烹饪，健康又美味。",
                "score": 4.7,
                "category": "调味品"
            },
            {
                "product_name": "天然蜂蜜",
                "review_text": "蜂蜜质量很好，味道香甜，包装也很用心，会继续回购的。",
                "score": 4.9,
                "category": "食品"
            },
            {
                "product_name": "有机茶叶",
                "review_text": "茶叶香气扑鼻，冲泡后汤色清澈，口感醇厚，回甘持久。",
                "score": 4.6,
                "category": "饮料"
            },
            {
                "product_name": "全麦面包",
                "review_text": "面包新鲜松软，麦香浓郁，无添加剂，健康美味，早餐首选。",
                "score": 4.4,
                "category": "食品"
            },
            {
                "product_name": "有机坚果",
                "review_text": "坚果新鲜香脆，种类丰富，营养价值高，是健康零食的好选择。",
                "score": 4.8,
                "category": "零食"
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def create_embeddings_with_cache(self, texts: List[str]) -> List[List[float]]:
        """创建嵌入并缓存结果"""
        # 检查缓存
        new_texts = [t for t in texts if t not in self.embedding_cache]
        
        if new_texts:
            print(f"🔄 正在处理{len(new_texts)}个新文本...")
            new_embeddings = self.generate_embeddings(new_texts)
            
            for text, embedding in zip(new_texts, new_embeddings):
                self.embedding_cache[text] = embedding
            
            # 保存缓存
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        
        return [self.embedding_cache[text] for text in texts]
    
    def load_embeddings_from_cache(self) -> bool:
        """从缓存加载嵌入"""
        try:
            with open(self.cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            print(f"✅ 已加载{len(self.embedding_cache)}个缓存嵌入")
            return True
        except FileNotFoundError:
            print("🆕 未找到缓存，将创建新的嵌入")
            return False
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search_reviews(self, query: str, n: int = 3, min_score: float = 0.0) -> pd.DataFrame:
        """搜索相关评论"""
        if self.reviews_df is None:
            print("❌ 请先加载数据")
            return pd.DataFrame()
        
        # 生成查询嵌入
        query_embedding = self.generate_embeddings([query])[0]
        
        # 计算相似度
        similarities = []
        for embedding in self.reviews_df['embedding']:
            score = self.cosine_similarity(query_embedding, embedding)
            similarities.append(score)
        
        self.reviews_df['similarity'] = similarities
        
        # 过滤并排序
        results = self.reviews_df[
            self.reviews_df['similarity'] >= min_score
        ].sort_values('similarity', ascending=False).head(n)
        
        return results
    
    def recommend_similar_products(self, product_name: str, n: int = 3) -> pd.DataFrame:
        """推荐相似产品"""
        if self.reviews_df is None:
            print("❌ 请先加载数据")
            return pd.DataFrame()
        
        # 找到目标产品的评论
        target_reviews = self.reviews_df[
            self.reviews_df['product_name'].str.contains(product_name, case=False, na=False)
        ]
        
        if target_reviews.empty:
            print(f"❌ 未找到产品：{product_name}")
            return pd.DataFrame()
        
        # 使用目标产品的平均嵌入作为查询
        target_embedding = np.mean(target_reviews['embedding'].tolist(), axis=0)
        
        # 计算所有产品的相似度
        similarities = []
        for embedding in self.reviews_df['embedding']:
            score = self.cosine_similarity(target_embedding, embedding)
            similarities.append(score)
        
        self.reviews_df['similarity_to_target'] = similarities
        
        # 排除目标产品，推荐相似产品
        recommendations = self.reviews_df[
            ~self.reviews_df['product_name'].str.contains(product_name, case=False, na=False)
        ].sort_values('similarity_to_target', ascending=False).head(n)
        
        return recommendations
    
    def setup_system(self) -> bool:
        """设置系统"""
        # 加载数据
        self.reviews_df = self.load_sample_data()
        
        # 尝试从缓存加载
        cache_loaded = self.load_embeddings_from_cache()
        
        if not cache_loaded:
            # 创建新的嵌入
            texts = self.reviews_df['review_text'].tolist()
            embeddings = self.create_embeddings_with_cache(texts)
            self.reviews_df['embedding'] = embeddings
        else:
            # 从缓存加载嵌入
            texts = self.reviews_df['review_text'].tolist()
            embeddings = [self.embedding_cache[text] for text in texts]
            self.reviews_df['embedding'] = embeddings
        
        return True
    
    def demo_search_functionality(self):
        """演示搜索功能"""
        print("🚀 产品评论语义搜索演示")
        print("=" * 50)
        
        # 设置系统
        if not self.setup_system():
            return
        
        # 演示搜索
        test_queries = [
            "健康有机食品",
            "口感醇厚的饮料",
            "营养丰富的早餐",
            "天然无添加",
            "适合送礼的产品"
        ]
        
        for query in test_queries:
            print(f"\n🔍 搜索：'{query}'")
            results = self.search_reviews(query, n=2)
            
            if not results.empty:
                for idx, row in results.iterrows():
                    print(f"   📦 {row['product_name']} (评分：{row['score']})")
                    print(f"   💬 {row['review_text']}")
                    print(f"   📊 相似度：{row['similarity']:.3f}")
                    print()
            else:
                print("   ❌ 未找到相关评论")
    
    def demo_recommendation_functionality(self):
        """演示推荐功能"""
        print("\n🎯 产品推荐演示")
        print("=" * 30)
        
        # 演示推荐
        target_products = ["蜂蜜", "咖啡", "燕麦"]
        
        for product in target_products:
            print(f"\n📦 基于"{product}"的推荐：")
            recommendations = self.recommend_similar_products(product, n=2)
            
            if not recommendations.empty:
                for idx, row in recommendations.iterrows():
                    print(f"   🔸 {row['product_name']} (评分：{row['score']})")
                    print(f"      相似度：{row['similarity_to_target']:.3f}")
            else:
                print("   ❌ 暂无推荐")

def demo_real_world_scenario():
    """演示真实场景"""
    print("🛒 真实场景演示：电商评论分析")
    print("=" * 40)
    
    # 创建系统
    system = ProductReviewSearchSystem()
    
    # 运行演示
    system.demo_search_functionality()
    system.demo_recommendation_functionality()
    
    print("\n🎉 演示完成！")
    print("\n使用提示：")
    print("1. 搜索功能：输入任何产品描述，找到相关评论")
    print("2. 推荐功能：输入产品名称，找到相似产品")
    print("3. 缓存机制：避免重复计算，提升效率")
    print("4. 可扩展：可加载真实CSV数据")

def main():
    """主函数"""
    print("🚀 产品评论语义搜索系统")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        demo_real_world_scenario()
    except Exception as e:
        print(f"❌ 运行错误：{e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本嵌入(Embedding)全面学习教程
================================

什么是文本嵌入？
文本嵌入是将文本(词语、句子、文档)转换为固定长度的数值向量的技术。
这些向量能够捕捉文本的语义信息，使得语义相似的文本在向量空间中距离相近。

用途：
1. 语义搜索 - 理解查询意图，返回相关结果
2. 文本相似度计算 - 判断两段文本的相似程度
3. 推荐系统 - 基于内容相似性推荐
4. 聚类分析 - 将相似文本分组
5. 异常检测 - 识别与常规内容不符的文本
6. 情感分析 - 将文本映射到情感空间
7. 问答系统 - 找到与用户问题最匹配的答案
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from datetime import datetime
import pickle

class EmbeddingTutorial:
    def __init__(self, api_key: str = None):
        """初始化嵌入客户端"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v4"
        self.dimensions = 1024
        
    def get_embedding(self, text: str) -> List[float]:
        """获取单段文本的嵌入向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return [data.embedding for data in response.data]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def find_similar_texts(self, query: str, texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """找到与查询最相似的文本"""
        query_embedding = self.get_embedding(query)
        text_embeddings = self.get_embeddings_batch(texts)
        
        similarities = []
        for text, embedding in zip(texts, text_embeddings):
            sim = self.cosine_similarity(query_embedding, embedding)
            similarities.append((text, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def demo_semantic_search():
    """演示语义搜索功能"""
    print("🎯 语义搜索演示")
    print("=" * 50)
    
    # 示例文档库
    documents = [
        "苹果手机最新款iPhone 15发布了，配备A17芯片",
        "华为Mate 60系列搭载麒麟9000S处理器，支持5G网络",
        "小米14系列首发骁龙8 Gen3，性能大幅提升",
        "特斯拉Model Y降价2万，电动车市场竞争激烈",
        "比亚迪海豹DM-i混动版本即将上市，续航超1300公里",
        "星巴克推出秋季限定饮品，南瓜拿铁回归",
        "茅台酒价格上涨，白酒市场持续升温",
        "ChatGPT推出语音对话功能，AI助手更加智能"
    ]
    
    tutorial = EmbeddingTutorial()
    
    # 搜索查询
    queries = ["手机新品", "电动车", "人工智能", "咖啡饮品"]
    
    for query in queries:
        print(f"\n🔍 查询: '{query}'")
        results = tutorial.find_similar_texts(query, documents, top_k=2)
        for text, score in results:
            print(f"  📄 {text} (相似度: {score:.3f})")

def demo_text_classification():
    """演示文本分类"""
    print("\n🏷️ 文本分类演示")
    print("=" * 50)
    
    # 预定义类别
    categories = {
        "科技": ["人工智能突破", "新款芯片发布", "操作系统升级"],
        "财经": ["股市大涨", "央行降息", "企业财报"],
        "娱乐": ["电影上映", "明星八卦", "音乐专辑"],
        "体育": ["足球比赛", "篮球联赛", "奥运会"]
    }
    
    tutorial = EmbeddingTutorial()
    
    # 获取每个类别的中心向量
    category_centers = {}
    for category, examples in categories.items():
        embeddings = tutorial.get_embeddings_batch(examples)
        center = np.mean(embeddings, axis=0)
        category_centers[category] = center
    
    # 测试文本
    test_texts = [
        "谷歌发布最新AI模型Gemini",
        "美联储加息影响全球股市",
        "NBA总决赛即将打响",
        "新电影票房破10亿"
    ]
    
    for text in test_texts:
        text_embedding = np.array(tutorial.get_embedding(text))
        
        best_category = None
        best_score = -1
        
        for category, center in category_centers.items():
            similarity = tutorial.cosine_similarity(text_embedding.tolist(), center.tolist())
            if similarity > best_score:
                best_score = similarity
                best_category = category
        
        print(f"📄 '{text}' -> {best_category} (置信度: {best_score:.3f})")

def demo_recommendation_system():
    """演示推荐系统"""
    print("\n🎯 基于内容的推荐演示")
    print("=" * 50)
    
    # 用户阅读历史
    user_history = [
        "深度学习革命：神经网络的发展历程",
        "机器学习入门：从零开始理解算法",
        "人工智能的未来趋势分析",
        "数据科学家必备技能指南"
    ]
    
    # 待推荐文章
    articles = [
        "神经网络在图像识别中的应用案例",
        "传统统计学与机器学习的区别",
        "区块链技术如何改变金融行业",
        "Python数据分析实战教程",
        "云计算服务比较：AWS vs Azure vs GCP",
        "深度学习框架比较：TensorFlow vs PyTorch",
        "网络安全威胁与防护措施",
        "人工智能伦理问题探讨"
    ]
    
    tutorial = EmbeddingTutorial()
    
    # 获取用户兴趣向量
    history_embeddings = tutorial.get_embeddings_batch(user_history)
    user_interest = np.mean(history_embeddings, axis=0)
    
    # 计算推荐分数
    article_embeddings = tutorial.get_embeddings_batch(articles)
    recommendations = []
    
    for article, embedding in zip(articles, article_embeddings):
        score = tutorial.cosine_similarity(user_interest.tolist(), embedding)
        recommendations.append((article, score))
    
    # 排序并显示前5个推荐
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print("为您推荐的文章：")
    for i, (article, score) in enumerate(recommendations[:5], 1):
        print(f"{i}. {article} (相关度: {score:.3f})")

def demo_sentiment_analysis():
    """演示情感分析"""
    print("\n😊 情感分析演示")
    print("=" * 50)
    
    # 情感基准文本
    positive_examples = [
        "太棒了！这个产品非常好用",
        "服务很贴心，体验极佳",
        "非常满意，强烈推荐"
    ]
    
    negative_examples = [
        "很差劲，完全不值这个价格",
        "服务态度恶劣，让人失望",
        "产品质量有问题，不推荐购买"
    ]
    
    tutorial = EmbeddingTutorial()
    
    # 获取情感基准向量
    pos_embeddings = tutorial.get_embeddings_batch(positive_examples)
    neg_embeddings = tutorial.get_embeddings_batch(negative_examples)
    
    pos_center = np.mean(pos_embeddings, axis=0)
    neg_center = np.mean(neg_embeddings, axis=0)
    
    # 测试评论
    reviews = [
        "这个产品真的很好用，物超所值！",
        "一般般，没什么特别的",
        "质量太差了，后悔购买",
        "客服很耐心，问题解决了",
        "价格有点贵，但效果还可以"
    ]
    
    for review in reviews:
        embedding = np.array(tutorial.get_embedding(review))
        
        pos_sim = tutorial.cosine_similarity(embedding.tolist(), pos_center.tolist())
        neg_sim = tutorial.cosine_similarity(embedding.tolist(), neg_center.tolist())
        
        if pos_sim > neg_sim:
            sentiment = "正面"
            confidence = pos_sim
        else:
            sentiment = "负面"
            confidence = neg_sim
        
        print(f"📄 '{review}' -> {sentiment}情感 (置信度: {confidence:.3f})")

def demo_clustering():
    """演示文本聚类"""
    print("\n🎯 文本聚类演示")
    print("=" * 50)
    
    from sklearn.cluster import KMeans
    
    # 混合文本
    texts = [
        "iPhone 15发布，性能大幅提升",
        "华为Mate 60支持卫星通信功能",
        "特斯拉Model 3价格降至20万以内",
        "比亚迪新能源车销量创新高",
        "周杰伦新专辑即将发行",
        "Taylor Swift世界巡演开始",
        "股市大涨，投资者信心增强",
        "央行宣布降息政策"
    ]
    
    tutorial = EmbeddingTutorial()
    embeddings = tutorial.get_embeddings_batch(texts)
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # 显示聚类结果
    cluster_groups = {0: [], 1: [], 2: []}
    for text, cluster in zip(texts, clusters):
        cluster_groups[cluster].append(text)
    
    for cluster_id, group_texts in cluster_groups.items():
        print(f"\n聚类 {cluster_id + 1}:")
        for text in group_texts:
            print(f"  📄 {text}")

def save_and_load_embeddings():
    """演示嵌入向量的保存和加载"""
    print("\n💾 嵌入向量保存与加载")
    print("=" * 50)
    
    tutorial = EmbeddingTutorial()
    
    # 创建知识库
    knowledge_base = {
        "什么是机器学习": "机器学习是人工智能的一个分支，让计算机通过数据学习",
        "深度学习": "深度学习是机器学习的一种方法，使用神经网络处理复杂问题",
        "自然语言处理": "NLP让计算机理解和处理人类语言",
        "计算机视觉": "让计算机能够理解和分析图像和视频内容"
    }
    
    # 生成并保存嵌入
    embeddings = {}
    for title, content in knowledge_base.items():
        embedding = tutorial.get_embedding(content)
        embeddings[title] = {
            "content": content,
            "embedding": embedding
        }
    
    # 保存到文件
    filename = f"knowledge_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)
    
    print(f"✅ 知识库嵌入已保存到: {filename}")
    
    # 从文件加载
    with open(filename, 'rb') as f:
        loaded_embeddings = pickle.load(f)
    
    print(f"📊 已加载 {len(loaded_embeddings)} 个知识条目")
    
    # 使用加载的嵌入进行搜索
    query = "什么是AI"
    results = []
    
    query_embedding = tutorial.get_embedding(query)
    for title, data in loaded_embeddings.items():
        similarity = tutorial.cosine_similarity(query_embedding, data["embedding"])
        results.append((title, data["content"], similarity))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🔍 查询: '{query}'")
    for title, content, score in results[:2]:
        print(f"📄 {title} (相似度: {score:.3f})")
        print(f"   {content}")

def main():
    """主函数：运行所有演示"""
    print("🚀 文本嵌入(Embedding)全面学习教程")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return
    
    try:
        # 运行各种演示
        demo_semantic_search()
        demo_text_classification()
        demo_recommendation_system()
        demo_sentiment_analysis()
        demo_clustering()
        save_and_load_embeddings()
        
        print("\n🎉 教程完成！文本嵌入的应用场景包括：")
        print("   • 语义搜索和问答系统")
        print("   • 文本分类和情感分析")
        print("   • 个性化推荐")
        print("   • 内容聚类和去重")
        print("   • 异常检测和垃圾内容过滤")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级文本嵌入应用示例
====================

本文件展示文本嵌入在实际业务中的高级应用
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from datetime import datetime
import sqlite3

class AdvancedEmbeddingSystem:
    """高级文本嵌入系统"""
    
    def __init__(self, api_key: str = None, db_path: str = "embeddings.db"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v4"
        self.dimensions = 1024
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_document(self, text: str, metadata: Dict = None) -> int:
        """添加文档到数据库"""
        embedding = self.get_embedding(text)
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (text, embedding, metadata) VALUES (?, ?, ?)",
            (text, embedding_str, json.dumps(metadata or {}))
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id
    
    def search_similar(self, query: str, limit: int = 5, threshold: float = 0.7) -> List[Dict]:
        """在数据库中搜索相似文档"""
        query_embedding = self.get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, embedding, metadata FROM documents")
        
        results = []
        for row in cursor.fetchall():
            doc_id, text, embedding_str, metadata = row
            doc_embedding = json.loads(embedding_str)
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= threshold:
                results.append({
                    'id': doc_id,
                    'text': text,
                    'similarity': similarity,
                    'metadata': json.loads(metadata)
                })
        
        conn.close()
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:limit]
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        # API限制批处理大小为10
        max_batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i+max_batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            all_embeddings.extend([data.embedding for data in response.data])
        
        return all_embeddings
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def visualize_embeddings(self, texts: List[str], labels: List[str] = None, 
                           method: str = "tsne", save_path: str = None):
        """可视化文本嵌入"""
        # 限制批处理大小为10以内
        batch_size = 10
        if len(texts) > batch_size:
            print(f"⚠️ 文本数量过多({len(texts)}个)，限制为{batch_size}个")
            texts = texts[:batch_size]
            if labels:
                labels = labels[:batch_size]
        
        embeddings = self.get_embeddings_batch(texts)
        embeddings_array = np.array(embeddings)
        
        # 降维
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(texts)-1))
        
        reduced_embeddings = reducer.fit_transform(embeddings_array)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                            alpha=0.6, s=100)
        
        # 添加标签 - 使用英文标签避免中文显示问题
        fallback_labels = [
            "ML", "DL", "NN", "Python", "Java", "Web", 
            "Food", "Pasta", "Sushi", "Sports"
        ]
        
        for i, text in enumerate(texts):
            if labels and i < len(labels):
                label = labels[i]
                # 如果标签是中文，使用英文fallback
                if any('\u4e00' <= c <= '\u9fff' for c in str(label)):
                    label = fallback_labels[i] if i < len(fallback_labels) else f"Item{i+1}"
            else:
                # 中文文本使用英文缩写
                label = fallback_labels[i] if i < len(fallback_labels) else f"Item{i+1}"
            
            ax.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), 
                        fontsize=10, alpha=0.8, ha='center')
        
        ax.set_title(f"Text Embedding Visualization ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)
        
        # 添加文本说明
        fig.text(0.02, 0.02, 
                f"Texts: {len(texts)} | Method: {method.upper()} | Dimension: {self.dimensions}", 
                fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化已保存为 {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def build_knowledge_base(self, documents: List[Dict]):
        """构建知识库"""
        print("🔨 构建知识库...")
        for doc in documents:
            doc_id = self.add_document(
                doc['text'], 
                doc.get('metadata', {})
            )
            print(f"✅ 已添加文档 {doc_id}: {doc['text'][:50]}...")
    
    def smart_qa_system(self, question: str, context_docs: int = 3) -> Dict:
        """智能问答系统"""
        similar_docs = self.search_similar(question, limit=context_docs)
        
        if not similar_docs:
            return {
                "answer": "抱歉，没有找到相关信息",
                "confidence": 0.0,
                "sources": []
            }
        
        # 简单的问题回答（实际应用中可以使用更复杂的LLM）
        context = "\n".join([doc['text'] for doc in similar_docs])
        
        return {
            "answer": f"基于找到的相关信息，可能的答案是：{similar_docs[0]['text'][:100]}...",
            "confidence": similar_docs[0]['similarity'],
            "sources": similar_docs
        }

def demo_knowledge_base():
    """演示知识库系统"""
    print("📚 知识库系统演示")
    print("=" * 50)
    
    system = AdvancedEmbeddingSystem()
    
    # 构建知识库
    documents = [
        {
            "text": "机器学习是一种人工智能方法，让计算机从数据中学习模式",
            "metadata": {"category": "技术", "tags": ["AI", "基础"]}
        },
        {
            "text": "深度学习是机器学习的一个分支，使用神经网络处理复杂任务",
            "metadata": {"category": "技术", "tags": ["AI", "深度学习"]}
        },
        {
            "text": "Python是最流行的数据科学编程语言，有丰富的库支持",
            "metadata": {"category": "编程", "tags": ["Python", "数据科学"]}
        },
        {
            "text": "卷积神经网络(CNN)主要用于图像识别和处理任务",
            "metadata": {"category": "技术", "tags": ["深度学习", "CNN"]}
        },
        {
            "text": "循环神经网络(RNN)适用于处理序列数据，如文本和时间序列",
            "metadata": {"category": "技术", "tags": ["深度学习", "RNN"]}
        }
    ]
    
    system.build_knowledge_base(documents)
    
    # 问答测试
    questions = [
        "什么是机器学习？",
        "CNN和RNN有什么区别？",
        "Python在数据科学中的应用"
    ]
    
    for question in questions:
        print(f"\n❓ 问题: {question}")
        result = system.smart_qa_system(question)
        print(f"🤖 回答: {result['answer']}")
        print(f"📊 置信度: {result['confidence']:.3f}")

def demo_anomaly_detection():
    """演示异常检测"""
    print("\n🔍 异常检测演示")
    print("=" * 50)
    
    system = AdvancedEmbeddingSystem()
    
    # 正常评论
    normal_reviews = [
        "这个产品质量很好，我很满意",
        "物流速度快，包装完好",
        "客服态度很好，解决问题及时",
        "价格合理，物有所值",
        "会再次购买，推荐给大家"
    ]
    
    # 异常评论（可能包含广告、垃圾信息等）
    suspicious_reviews = [
        "🔥🔥🔥限时抢购！点击链接获取优惠🔥🔥🔥",
        "加我微信：XXXXX，获取更多优惠信息",
        "这是一个测试评论，没有任何意义的内容",
        "垃圾垃圾垃圾垃圾垃圾垃圾垃圾垃圾",
        "请联系QQ：123456789，专业刷单团队"
    ]
    
    # 计算正常评论的中心向量
    normal_embeddings = system.get_embeddings_batch(normal_reviews)
    normal_center = np.mean(normal_embeddings, axis=0)
    
    # 检测异常
    all_reviews = normal_reviews + suspicious_reviews
    all_embeddings = system.get_embeddings_batch(all_reviews)
    
    threshold = 0.5  # 异常阈值
    
    for review, embedding in zip(all_reviews, all_embeddings):
        similarity = system.cosine_similarity(embedding, normal_center.tolist())
        
        if similarity < threshold:
            print(f"⚠️ 异常: {review} (相似度: {similarity:.3f})")
        else:
            print(f"✅ 正常: {review[:20]}... (相似度: {similarity:.3f})")

def demo_semantic_search_engine():
    """演示语义搜索引擎"""
    print("\n🔎 语义搜索引擎演示")
    print("=" * 50)
    
    system = AdvancedEmbeddingSystem()
    
    # 电商产品描述
    products = [
        "iPhone 15 Pro Max 256GB 原色钛金属 5G手机",
        "华为Mate 60 Pro 12GB+512GB 雅黑 卫星通信",
        "小米14 Ultra 16GB+1TB 钛金属版 徕卡影像",
        "MacBook Pro M3 14英寸 18GB+512GB 深空黑",
        "华为MateBook X Pro 13代酷睿 32GB+1TB",
        "戴森V12无线吸尘器 手持除螨 激光探测",
        "iPad Pro M2 12.9英寸 256GB WiFi版",
        "索尼WH-1000XM5 降噪耳机 无线蓝牙"
    ]
    
    # 添加到知识库
    for product in products:
        system.add_document(product, {"type": "product", "category": "电子产品"})
    
    # 用户搜索查询
    search_queries = [
        "最好的拍照手机",
        "办公用的笔记本电脑",
        "苹果的产品",
        "无线耳机",
        "高端手机"
    ]
    
    for query in search_queries:
        print(f"\n🔍 搜索: '{query}'")
        results = system.search_similar(query, limit=3)
        
        for result in results:
            print(f"  📱 {result['text']} (相关度: {result['similarity']:.3f})")

def demo_visualization():
    """演示可视化"""
    print("\n🎨 文本嵌入可视化演示")
    print("=" * 50)
    
    system = AdvancedEmbeddingSystem()
    
    # 准备数据 - 限制为10个以内
    texts = [
        "机器学习算法",
        "深度学习模型",
        "神经网络",
        "Python编程",
        "Java开发",
        "Web前端",
        "牛排烹饪",
        "意大利面",
        "寿司制作",
        "足球比赛"
    ]
    
    labels = ["AI", "AI", "AI", "编程", "编程", "编程", 
              "美食", "美食", "美食", "运动"]
    
    # 创建可视化
    system.visualize_embeddings(texts, labels, method="tsne", 
                               save_path="embeddings_visualization.png")
    print("✅ 可视化已保存为 embeddings_visualization.png")

def main():
    """主函数"""
    print("🚀 高级文本嵌入应用演示")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        return
    
    try:
        # 运行高级演示
        demo_knowledge_base()
        demo_anomaly_detection()
        demo_semantic_search_engine()
        demo_visualization()
        
        print("\n🎉 高级应用演示完成！")
        print("这些技术可以应用于：")
        print("   • 智能客服系统")
        print("   • 内容推荐引擎")
        print("   • 垃圾内容过滤")
        print("   • 语义搜索引擎")
        print("   • 知识图谱构建")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
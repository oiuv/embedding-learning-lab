#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战项目3：语义搜索引擎
==================

基于文本嵌入的智能搜索引擎，实现语义理解、搜索结果聚类、个性化排序。

项目功能：
1. 自然语言查询理解
2. 语义搜索匹配
3. 搜索结果聚类
4. 个性化排序
5. 搜索建议
6. 搜索分析

技术栈：
- 文本嵌入：text-embedding-v4
- 向量索引：FAISS
- 聚类算法：K-means
- 排序算法：BM25 + 语义相似度
- Web框架：Flask
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime
import hashlib
from dataclasses import dataclass
import re
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    title: str
    content: str
    score: float
    category: str
    highlights: List[str]
    metadata: Dict

class SemanticSearchEngine:
    """语义搜索引擎"""
    
    def __init__(self, db_path: str = "search_engine.db"):
        """初始化搜索引擎"""
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        self.documents = {}
        self.search_cache = {}
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                url TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                view_count INTEGER DEFAULT 0
            )
        ''')
        
        # 搜索历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                user_id TEXT,
                results_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 搜索分析表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                clicked_doc_id TEXT,
                position INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc_id: str, title: str, content: str, category: str = None, tags: List[str] = None, url: str = None):
        """添加文档"""
        # 清理内容
        clean_content = self.clean_text(content)
        
        # 生成嵌入
        embedding = self.client.get_embedding(f"{title} {clean_content}")
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, title, content, category, tags, url, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, title, clean_content, category, json.dumps(tags or []), url, embedding_str))
        
        conn.commit()
        conn.close()
        
        # 更新内存缓存
        self.documents[doc_id] = {
            'title': title,
            'content': clean_content,
            'category': category,
            'tags': tags or [],
            'url': url,
            'embedding': embedding
        }
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 合并空格
        text = ' '.join(text.split())
        return text
    
    def semantic_search(self, query: str, user_id: str = None, category: str = None, limit: int = 10) -> List[SearchResult]:
        """语义搜索"""
        # 生成查询嵌入
        query_embedding = self.client.get_embedding(query)
        
        # 缓存搜索
        cache_key = hashlib.md5(f"{query}_{category}_{limit}".encode()).hexdigest()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询
        base_query = '''
            SELECT id, title, content, category, tags, url, view_count
            FROM documents
        '''
        params = []
        
        if category:
            base_query += ' WHERE category = ?'
            params.append(category)
        
        cursor.execute(base_query, params)
        docs = cursor.fetchall()
        conn.close()
        
        # 计算语义相似度
        results = []
        query_vec = np.array(query_embedding)
        
        for doc in docs:
            doc_id, title, content, category, tags_str, url, view_count = doc
            
            # 获取文档嵌入
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT embedding FROM documents WHERE id = ?', (doc_id,))
            embedding_str = cursor.fetchone()[0]
            conn.close()
            
            doc_vec = np.array(json.loads(embedding_str))
            
            # 计算语义相似度
            semantic_score = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            
            # 关键词匹配分数
            keyword_score = self.calculate_keyword_score(query, title, content)
            
            # 综合分数
            final_score = 0.7 * semantic_score + 0.3 * keyword_score
            
            # 生成高亮
            highlights = self.generate_highlights(query, title, content)
            
            results.append(SearchResult(
                doc_id=doc_id,
                title=title,
                content=content[:200] + '...' if len(content) > 200 else content,
                score=final_score,
                category=category,
                highlights=highlights,
                metadata={
                    'url': url,
                    'view_count': view_count,
                    'tags': json.loads(tags_str)
                }
            ))
        
        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 记录搜索历史
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO search_history (query, user_id, results_count)
            VALUES (?, ?, ?)
        ''', (query, user_id, len(results)))
        conn.commit()
        conn.close()
        
        # 缓存结果
        self.search_cache[cache_key] = results[:limit]
        
        return results[:limit]
    
    def calculate_keyword_score(self, query: str, title: str, content: str) -> float:
        """计算关键词匹配分数"""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        
        # 标题匹配
        title_match = len(query_words & title_words) / len(query_words) if query_words else 0
        
        # 内容匹配
        content_match = len(query_words & content_words) / len(query_words) if query_words else 0
        
        return 0.6 * title_match + 0.4 * content_match
    
    def generate_highlights(self, query: str, title: str, content: str) -> List[str]:
        """生成搜索高亮"""
        query_words = query.lower().split()
        
        highlights = []
        
        # 标题高亮
        title_lower = title.lower()
        for word in query_words:
            if word in title_lower:
                start = title_lower.find(word)
                if start >= 0:
                    highlight = title[max(0, start-10):start+len(word)+10]
                    highlights.append(highlight)
        
        # 内容高亮
        content_lower = content.lower()
        for word in query_words:
            if word in content_lower:
                positions = [i for i in range(len(content_lower)) 
                           if content_lower.startswith(word, i)]
                for pos in positions[:2]:  # 最多2个高亮
                    highlight = content[max(0, pos-20):pos+len(word)+20]
                    highlights.append(highlight)
        
        return highlights[:3]  # 最多3个高亮
    
    def cluster_search_results(self, results: List[SearchResult], n_clusters: int = 3) -> Dict:
        """聚类搜索结果"""
        if len(results) < n_clusters:
            return {'clusters': [], 'noise': results}
        
        # 提取嵌入向量
        embeddings = []
        for result in results:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT embedding FROM documents WHERE id = ?', (result.doc_id,))
            embedding_str = cursor.fetchone()[0]
            conn.close()
            
            embeddings.append(json.loads(embedding_str))
        
        embeddings_array = np.array(embeddings)
        
        # 使用K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        
        # 构建聚类结果
        clusters = [[] for _ in range(n_clusters)]
        noise = []
        
        for i, (result, label) in enumerate(zip(results, labels)):
            clusters[label].append(result)
        
        # 为每个聚类生成主题
        cluster_topics = []
        for i, cluster in enumerate(clusters):
            if cluster:
                # 提取关键词作为主题
                all_text = ' '.join([r.title + ' ' + r.content for r in cluster])
                words = re.findall(r'\w+', all_text.lower())
                word_counts = Counter(words)
                top_words = [w for w, c in word_counts.most_common(5) if len(w) > 2]
                
                cluster_topics.append({
                    'cluster_id': i,
                    'topic': ' '.join(top_words),
                    'documents': cluster,
                    'size': len(cluster)
                })
        
        return {'clusters': cluster_topics, 'noise': noise}
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """获取搜索建议"""
        # 基于历史搜索和文档内容生成建议
        suggestions = []
        
        # 获取相似的历史查询
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT query FROM search_history
            WHERE query LIKE ? AND LENGTH(query) > 2
            ORDER BY timestamp DESC
            LIMIT 10
        ''', (f"{query}%",))
        
        history_queries = [row[0] for row in cursor.fetchall()]
        suggestions.extend(history_queries)
        
        # 基于文档标题生成建议
        cursor.execute('''
            SELECT DISTINCT title FROM documents
            WHERE title LIKE ?
            ORDER BY view_count DESC
            LIMIT 10
        ''', (f"%{query}%",))
        
        title_suggestions = [row[0] for row in cursor.fetchall()]
        suggestions.extend(title_suggestions)
        
        conn.close()
        
        # 去重并限制数量
        unique_suggestions = list(set(suggestions))
        return unique_suggestions[:limit]
    
    def record_click(self, query: str, doc_id: str, position: int, user_id: str = None):
        """记录点击行为"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_analytics (query, clicked_doc_id, position)
            VALUES (?, ?, ?)
        ''', (query, doc_id, position))
        
        # 更新文档查看计数
        cursor.execute('''
            UPDATE documents SET view_count = view_count + 1 WHERE id = ?
        ''', (doc_id,))
        
        conn.commit()
        conn.close()
    
    def get_search_analytics(self) -> Dict:
        """获取搜索分析数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本统计
        cursor.execute('SELECT COUNT(*) FROM search_history')
        total_searches = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM search_analytics')
        total_clicks = cursor.fetchone()[0]
        
        # 热门查询
        cursor.execute('''
            SELECT query, COUNT(*) as count
            FROM search_history
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
        ''')
        top_queries = cursor.fetchall()
        
        # 点击率
        ctr = total_clicks / total_searches if total_searches > 0 else 0
        
        conn.close()
        
        return {
            'total_searches': total_searches,
            'total_clicks': total_clicks,
            'ctr': ctr,
            'top_queries': top_queries
        }
    
    def load_sample_documents(self):
        """加载示例文档"""
        sample_docs = [
            {
                'doc_id': 'doc_001',
                'title': '机器学习入门教程',
                'content': '机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。主要类型包括监督学习、无监督学习和强化学习。',
                'category': '技术',
                'tags': ['机器学习', 'AI', '教程'],
                'url': '/docs/ml-intro'
            },
            {
                'doc_id': 'doc_002',
                'title': '深度学习实战指南',
                'content': '深度学习使用多层神经网络解决复杂问题。需要大量数据和计算资源，在图像识别、自然语言处理等领域表现出色。',
                'category': '技术',
                'tags': ['深度学习', '神经网络', '实战'],
                'url': '/docs/deep-learning-guide'
            },
            {
                'doc_id': 'doc_003',
                'title': 'Python数据分析完全指南',
                'content': 'Python是数据分析的首选语言，主要使用Pandas、NumPy、Matplotlib等库。本指南涵盖从基础到高级的数据分析技巧。',
                'category': '编程',
                'tags': ['Python', '数据分析', 'Pandas'],
                'url': '/docs/python-data-analysis'
            },
            {
                'doc_id': 'doc_004',
                'title': '区块链技术原理与应用',
                'content': '区块链是一种分布式账本技术，通过密码学保证数据安全。应用包括加密货币、智能合约、供应链管理等。',
                'category': '技术',
                'tags': ['区块链', '分布式系统', '应用'],
                'url': '/docs/blockchain-principles'
            }
        ]
        
        for doc in sample_docs:
            self.add_document(
                doc['doc_id'],
                doc['title'],
                doc['content'],
                doc['category'],
                doc['tags'],
                doc['url']
            )
    
    def demo_search_engine(self):
        """演示搜索引擎"""
        print("🚀 实战项目3：语义搜索引擎")
        print("=" * 60)
        
        # 加载示例文档
        print("📚 加载示例文档...")
        self.load_sample_documents()
        
        # 演示搜索功能
        test_queries = [
            "机器学习",
            "深度学习教程",
            "Python数据分析",
            "区块链技术"
        ]
        
        print("\n🎯 语义搜索演示")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\n❓ 查询: {query}")
            results = self.semantic_search(query, limit=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. {result.title}")
                print(f"      分数: {result.score:.3f}")
                print(f"      内容: {result.content[:100]}...")
                print(f"      高亮: {result.highlights}")
        
        # 演示聚类功能
        print("\n🎯 搜索结果聚类演示")
        print("=" * 50)
        
        results = self.semantic_search("技术教程", limit=5)
        clusters = self.cluster_search_results(results, n_clusters=2)
        
        for cluster in clusters['clusters']:
            print(f"\n📊 聚类: {cluster['topic']}")
            print(f"   文档数: {cluster['size']}")
            for doc in cluster['documents']:
                print(f"   - {doc.title}")
        
        # 演示搜索建议
        print("\n🎯 搜索建议演示")
        print("=" * 50)
        
        suggestion_queries = ["机器", "深度", "Python"]
        for query in suggestion_queries:
            suggestions = self.get_search_suggestions(query)
            print(f"\n❓ 查询: {query}")
            print(f"   建议: {suggestions}")
        
        # 显示分析数据
        print("\n📊 搜索分析数据")
        print("=" * 50)
        
        analytics = self.get_search_analytics()
        print(f"📈 总搜索数: {analytics['total_searches']}")
        print(f"👆 总点击数: {analytics['total_clicks']}")
        print(f"📊 点击率: {analytics['ctr']:.2%}")
        
        if analytics['top_queries']:
            print("\n🔥 热门查询:")
            for query, count in analytics['top_queries']:
                print(f"   {query}: {count} 次")
        
        print("\n🎉 语义搜索引擎演示完成！")
        print("\n下一步：")
        print("   1. 启动搜索服务: python search_service.py")
        print("   2. 构建索引: python build_index.py")
        print("   3. 测试搜索API: python test_search.py")

def main():
    """主函数"""
    print("🚀 语义搜索引擎")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        search_engine = SemanticSearchEngine()
        search_engine.demo_search_engine()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级功能第1课：智能知识库系统
=========================

构建企业级智能知识库，实现语义查询、知识图谱、增量更新等功能。

学习目标：
1. 构建知识图谱和语义网络
2. 实现智能问答和查询优化
3. 设计增量更新机制
4. 多模态知识融合
5. 企业级知识管理系统
"""

import os
import sys
import json
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
import pickle
from datetime import datetime
import hashlib

# 修复Python 3.12 datetime警告
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("timestamp", lambda x: datetime.fromisoformat(x.decode()))

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

class KnowledgeBaseSystem:
    """智能知识库系统"""
    
    def __init__(self, db_path: str = "knowledge_base.db"):
        """初始化知识库"""
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        self.embedding_cache = {}
        
    def init_database(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 知识条目表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1
            )
        ''')
        
        # 知识关系表（知识图谱）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                relationship_type TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES knowledge_entries(id),
                FOREIGN KEY (target_id) REFERENCES knowledge_entries(id)
            )
        ''')
        
        # 查询历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT,
                relevance_score REAL,
                query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_knowledge(self, title: str, content: str, category: str = "", tags: List[str] = None) -> int:
        """添加知识条目"""
        embedding = self.client.get_embedding(f"{title} {content}")
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge_entries (title, content, category, tags, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, content, category, json.dumps(tags or []), embedding_str))
        
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ 已添加知识条目: {title} (ID: {entry_id})")
        return entry_id
    
    def build_knowledge_graph(self, entries: List[Dict]) -> Dict:
        """构建知识图谱"""
        print("🔨 构建知识图谱...")
        
        # 添加所有知识条目
        entry_ids = []
        for entry in entries:
            entry_id = self.add_knowledge(
                entry['title'],
                entry['content'],
                entry.get('category', ''),
                entry.get('tags', [])
            )
            entry_ids.append(entry_id)
        
        # 构建关系（基于语义相似度）
        relationships = self._build_relationships(entry_ids)
        
        return {
            'total_entries': len(entry_ids),
            'relationships': len(relationships),
            'entry_ids': entry_ids
        }
    
    def _build_relationships(self, entry_ids: List[int]) -> List[Dict]:
        """基于语义相似度构建知识关系"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取所有条目
        cursor.execute('SELECT id, title, content, embedding FROM knowledge_entries WHERE id IN ({})'.format(
            ','.join('?' * len(entry_ids))), entry_ids)
        entries = cursor.fetchall()
        
        relationships = []
        
        # 计算条目间的语义相似度
        for i, (id1, title1, content1, emb1_str) in enumerate(entries):
            embedding1 = np.array(json.loads(emb1_str))
            
            for j, (id2, title2, content2, emb2_str) in enumerate(entries[i+1:], i+1):
                embedding2 = np.array(json.loads(emb2_str))
                
                # 计算余弦相似度
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                
                if similarity > 0.7:  # 相似度阈值
                    relationship = {
                        'source_id': id1,
                        'target_id': id2,
                        'relationship_type': 'semantic_similarity',
                        'confidence': similarity
                    }
                    relationships.append(relationship)
                    
                    # 保存到数据库
                    cursor.execute('''
                        INSERT INTO knowledge_relationships (source_id, target_id, relationship_type, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (id1, id2, 'semantic_similarity', float(similarity)))
        
        conn.commit()
        conn.close()
        
        print(f"✅ 已构建 {len(relationships)} 个知识关系")
        return relationships
    
    def semantic_query(self, query: str, top_k: int = 5, category: str = None) -> List[Dict]:
        """语义查询"""
        query_embedding = self.client.get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询条件
        base_query = '''
            SELECT id, title, content, category, tags, created_at, embedding
            FROM knowledge_entries
        '''
        params = []
        
        if category:
            base_query += ' WHERE category = ?'
            params.append(category)
        
        cursor.execute(base_query, params)
        entries = cursor.fetchall()
        
        # 计算相似度并排序
        results = []
        for entry in entries:
            id, title, content, cat, tags_str, created_at, emb_str = entry
            embedding = np.array(json.loads(emb_str))
            
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            results.append({
                'id': id,
                'title': title,
                'content': content[:200] + '...' if len(content) > 200 else content,
                'category': cat,
                'tags': json.loads(tags_str),
                'similarity': similarity,
                'created_at': created_at
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:top_k]
        
        # 记录查询历史
        self._record_query(query, json.dumps(top_results), 
                          top_results[0]['similarity'] if top_results else 0)
        
        conn.close()
        return top_results
    
    def smart_qa(self, question: str) -> Dict:
        """智能问答"""
        # 语义查询获取相关知识
        relevant_knowledge = self.semantic_query(question, top_k=3)
        
        if not relevant_knowledge:
            return {
                'answer': '抱歉，没有找到相关知识点',
                'confidence': 0.0,
                'sources': []
            }
        
        # 构建答案（简化版，实际中可使用更复杂的LLM）
        context = '\n'.join([k['content'] for k in relevant_knowledge])
        
        # 基于最相关知识生成答案
        best_match = relevant_knowledge[0]
        
        answer = f"基于知识库，{best_match['title']}的相关内容是：\n{best_match['content']}"
        
        return {
            'answer': answer,
            'confidence': best_match['similarity'],
            'sources': relevant_knowledge
        }
    
    def incremental_update(self, new_entries: List[Dict]) -> Dict:
        """增量更新"""
        print("🔄 执行增量更新...")
        
        updated_count = 0
        new_count = 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for entry in new_entries:
            # 检查是否已存在（基于标题+内容哈希）
            content_hash = hashlib.md5(f"{entry['title']}{entry['content']}".encode()).hexdigest()
            
            cursor.execute('''
                SELECT id, content FROM knowledge_entries 
                WHERE title = ? AND content = ?
            ''', (entry['title'], entry['content']))
            
            existing = cursor.fetchone()
            
            if existing:
                # 更新现有条目
                cursor.execute('''
                    UPDATE knowledge_entries 
                    SET content = ?, category = ?, tags = ?, updated_at = ?, version = version + 1
                    WHERE id = ?
                ''', (entry['content'], entry.get('category', ''), 
                      json.dumps(entry.get('tags', [])), datetime.now(), existing[0]))
                updated_count += 1
            else:
                # 添加新条目
                self.add_knowledge(entry['title'], entry['content'], 
                                 entry.get('category', ''), entry.get('tags', []))
                new_count += 1
        
        conn.commit()
        conn.close()
        
        # 重新构建受影响的关系
        if new_count > 0 or updated_count > 0:
            self._rebuild_relationships()
        
        return {
            'new_entries': new_count,
            'updated_entries': updated_count,
            'total_entries': new_count + updated_count
        }
    
    def _rebuild_relationships(self):
        """重建知识关系"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清空现有关系
        cursor.execute('DELETE FROM knowledge_relationships')
        
        # 获取所有条目ID
        cursor.execute('SELECT id FROM knowledge_entries')
        entry_ids = [row[0] for row in cursor.fetchall()]
        
        conn.commit()
        conn.close()
        
        # 重新构建关系
        self._build_relationships(entry_ids)
    
    def _record_query(self, query: str, response: str, relevance_score: float):
        """记录查询历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO query_history (query, response, relevance_score)
            VALUES (?, ?, ?)
        ''', (query, response, relevance_score))
        
        conn.commit()
        conn.close()
    
    def get_analytics(self) -> Dict:
        """获取知识库分析数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本统计
        cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
        total_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM knowledge_relationships')
        total_relationships = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM query_history')
        total_queries = cursor.fetchone()[0]
        
        # 类别分布
        cursor.execute('SELECT category, COUNT(*) FROM knowledge_entries GROUP BY category')
        category_stats = dict(cursor.fetchall())
        
        # 查询统计
        cursor.execute('SELECT AVG(relevance_score) FROM query_history')
        avg_relevance = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'total_relationships': total_relationships,
            'total_queries': total_queries,
            'category_stats': category_stats,
            'average_relevance_score': avg_relevance
        }
    
    def load_sample_knowledge(self) -> List[Dict]:
        """加载示例知识数据"""
        sample_knowledge = [
            {
                "title": "机器学习基础概念",
                "content": "机器学习是人工智能的一个分支，专注于让计算机从数据中学习模式，而无需明确编程。主要类型包括监督学习、无监督学习和强化学习。",
                "category": "人工智能",
                "tags": ["机器学习", "AI基础", "监督学习", "无监督学习"]
            },
            {
                "title": "深度学习原理",
                "content": "深度学习使用多层神经网络处理复杂问题，通过反向传播算法优化网络权重。常见架构包括CNN、RNN和Transformer。",
                "category": "人工智能",
                "tags": ["深度学习", "神经网络", "CNN", "RNN", "Transformer"]
            },
            {
                "title": "Python数据科学工具链",
                "content": "Python在数据科学中广泛使用，主要工具包括NumPy、Pandas、Scikit-learn、Matplotlib和TensorFlow/PyTorch。",
                "category": "编程语言",
                "tags": ["Python", "数据科学", "NumPy", "Pandas", "Scikit-learn"]
            },
            {
                "title": "自然语言处理应用",
                "content": "NLP让计算机理解和处理人类语言，应用包括文本分类、情感分析、机器翻译和问答系统。",
                "category": "人工智能",
                "tags": ["NLP", "文本分类", "情感分析", "机器翻译", "问答系统"]
            },
            {
                "title": "云计算服务类型",
                "content": "云计算提供三种主要服务类型：IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。",
                "category": "技术架构",
                "tags": ["云计算", "IaaS", "PaaS", "SaaS", "架构"]
            },
            {
                "title": "区块链技术原理",
                "content": "区块链是分布式账本技术，通过密码学保证数据不可篡改，主要应用于加密货币、供应链管理和智能合约。",
                "category": "技术架构",
                "tags": ["区块链", "分布式账本", "加密货币", "智能合约"]
            }
        ]
        return sample_knowledge
    
    def demo_knowledge_base(self):
        """演示知识库系统"""
        print("🚀 高级功能第1课：智能知识库系统")
        print("=" * 60)
        
        # 加载示例知识
        sample_knowledge = self.load_sample_knowledge()
        
        print("📚 构建知识库...")
        stats = self.build_knowledge_graph(sample_knowledge)
        print(f"✅ 知识库构建完成：{stats['total_entries']} 条目，{stats['relationships']} 关系")
        
        # 演示查询功能
        print("\n🔍 第1部分：语义查询演示")
        print("=" * 50)
        
        test_queries = [
            "机器学习有哪些类型？",
            "Python数据分析工具",
            "区块链如何工作",
            "深度学习架构"
        ]
        
        for query in test_queries:
            print(f"\n❓ 查询: {query}")
            results = self.semantic_query(query, top_k=2)
            
            for result in results:
                print(f"   📖 {result['title']} (相似度: {result['similarity']:.3f})")
                print(f"      {result['content'][:100]}...")
        
        # 演示智能问答
        print("\n🤖 第2部分：智能问答演示")
        print("=" * 50)
        
        questions = [
            "什么是机器学习？",
            "Python在数据科学中有什么作用？",
            "区块链有哪些应用场景？"
        ]
        
        for question in questions:
            print(f"\n❓ 问题: {question}")
            response = self.smart_qa(question)
            print(f"   🤖 {response['answer'][:200]}...")
            print(f"   📊 置信度: {response['confidence']:.3f}")
        
        # 演示增量更新
        print("\n🔄 第3部分：增量更新演示")
        print("=" * 50)
        
        new_entries = [
            {
                "title": "强化学习简介",
                "content": "强化学习通过奖励和惩罚机制训练智能体做出最优决策，应用包括游戏AI和机器人控制。",
                "category": "人工智能",
                "tags": ["强化学习", "游戏AI", "机器人控制"]
            },
            {
                "title": "边缘计算概念",
                "content": "边缘计算将计算能力部署到数据源附近，减少延迟，适用于IoT和实时应用。",
                "category": "技术架构",
                "tags": ["边缘计算", "IoT", "实时处理"]
            }
        ]
        
        update_stats = self.incremental_update(new_entries)
        print(f"✅ 增量更新完成：新增 {update_stats['new_entries']} 条，更新 {update_stats['updated_entries']} 条")
        
        # 展示分析数据
        print("\n📊 第4部分：知识库分析")
        print("=" * 50)
        
        analytics = self.get_analytics()
        print(f"📈 总条目数: {analytics['total_entries']}")
        print(f"🔗 关系数: {analytics['total_relationships']}")
        print(f"❓ 查询数: {analytics['total_queries']}")
        print(f"🎯 平均相关度: {analytics['average_relevance_score']:.3f}")
        
        if analytics['category_stats']:
            print("\n📂 类别分布:")
            for category, count in analytics['category_stats'].items():
                print(f"   {category}: {count} 条")

def main():
    """主函数"""
    print("🚀 智能知识库系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        kb_system = KnowledgeBaseSystem()
        kb_system.demo_knowledge_base()
        
        print("\n🎉 知识库系统演示完成！")
        print("\n核心技术总结：")
        print("   • 语义查询和智能问答")
        print("   • 知识图谱构建")
        print("   • 增量更新机制")
        print("   • 多模态知识融合")
        print("\n实际应用场景：")
        print("   • 企业知识库")
        print("   • 技术文档问答")
        print("   • 产品手册智能查询")
        print("   • 教育知识管理系统")
        print("\n下一课：03-02-anomaly-detection.py - 异常检测")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战项目1：智能问答系统
==================

基于文本嵌入的智能问答系统，实现语义理解、知识检索、答案生成。

项目功能：
1. 自然语言问题理解
2. 语义知识检索
3. 精准答案匹配
4. 多轮对话支持
5. 知识库管理

技术栈：
- 文本嵌入：text-embedding-v4
- 向量存储：FAISS/Pinecone
- 问答匹配：语义相似度
- 后端：Flask REST API
- 前端：HTML/JavaScript
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
import sqlite3
from datetime import datetime
import hashlib
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

@dataclass
class QAItem:
    """问答条目"""
    question: str
    answer: str
    category: str = "general"
    tags: List[str] = None
    confidence: float = 0.0
    created_at: datetime = None

class SmartQASystem:
    """智能问答系统"""
    
    def __init__(self, db_path: str = "qa_system.db"):
        """初始化问答系统"""
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        self.embedding_cache = {}
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 问答知识表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                embedding TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                usage_count INTEGER DEFAULT 0
            )
        ''')
        
        # 对话历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_question TEXT,
                system_answer TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 用户反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qa_id INTEGER,
                feedback_type TEXT, -- 'good', 'bad', 'irrelevant'
                feedback_text TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (qa_id) REFERENCES qa_knowledge(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_qa_pair(self, question: str, answer: str, category: str = "general", tags: List[str] = None) -> int:
        """添加问答对"""
        # 计算问题嵌入
        embedding = self.client.get_embedding(question)
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO qa_knowledge (question, answer, category, tags, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', (question, answer, category, json.dumps(tags or []), embedding_str))
        
        qa_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"✅ 已添加问答对 (ID: {qa_id})")
        return qa_id
    
    def semantic_search(self, query: str, category: str = None, limit: int = 5) -> List[Dict]:
        """语义搜索相似问题"""
        query_embedding = self.client.get_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询
        base_query = '''
            SELECT id, question, answer, category, tags, confidence, created_at
            FROM qa_knowledge
        '''
        params = []
        
        if category:
            base_query += ' WHERE category = ?'
            params.append(category)
        
        cursor.execute(base_query, params)
        qa_pairs = cursor.fetchall()
        
        # 计算相似度
        results = []
        for qa in qa_pairs:
            id, question, answer, cat, tags_str, confidence, created_at = qa
            
            # 获取存储的嵌入
            cursor.execute('SELECT embedding FROM qa_knowledge WHERE id = ?', (id,))
            embedding_str = cursor.fetchone()[0]
            stored_embedding = np.array(json.loads(embedding_str))
            
            # 计算相似度
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            results.append({
                'id': id,
                'question': question,
                'answer': answer,
                'category': cat,
                'tags': json.loads(tags_str),
                'similarity': similarity,
                'confidence': confidence,
                'created_at': created_at
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:limit]
        
        conn.close()
        return top_results
    
    def answer_question(self, question: str, category: str = None, confidence_threshold: float = 0.7) -> Dict:
        """回答问题"""
        # 搜索相似问题
        similar_qas = self.semantic_search(question, category, limit=3)
        
        if not similar_qas:
            return {
                'question': question,
                'answer': "抱歉，没有找到相关答案",
                'confidence': 0.0,
                'sources': [],
                'category': category
            }
        
        # 获取最相似的答案
        best_match = similar_qas[0]
        
        if best_match['similarity'] < confidence_threshold:
            return {
                'question': question,
                'answer': "抱歉，没有找到足够相似的答案",
                'confidence': best_match['similarity'],
                'sources': similar_qas,
                'category': category
            }
        
        # 更新使用计数
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE qa_knowledge 
            SET usage_count = usage_count + 1, updated_at = ?
            WHERE id = ?
        ''', (datetime.now(), best_match['id']))
        conn.commit()
        conn.close()
        
        return {
            'question': question,
            'answer': best_match['answer'],
            'confidence': best_match['similarity'],
            'sources': similar_qas,
            'category': best_match['category'],
            'qa_id': best_match['id']
        }
    
    def multi_turn_conversation(self, session_id: str, user_input: str, context: List[Dict] = None) -> Dict:
        """多轮对话"""
        # 结合上下文理解问题
        if context:
            # 简单的上下文处理
            context_text = " ".join([f"用户: {c['user']}, 系统: {c['system']}" for c in context[-3:]])
            enhanced_query = f"{context_text} 当前问题: {user_input}"
        else:
            enhanced_query = user_input
        
        # 回答问题
        answer = self.answer_question(enhanced_query)
        
        # 记录对话历史
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_history (session_id, user_question, system_answer, confidence)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_input, answer['answer'], answer['confidence']))
        conn.commit()
        conn.close()
        
        return answer
    
    def load_sample_knowledge(self) -> List[QAItem]:
        """加载示例知识"""
        sample_qa = [
            QAItem(
                question="什么是机器学习？",
                answer="机器学习是人工智能的一个分支，让计算机从数据中学习模式，而无需明确编程。主要包括监督学习、无监督学习和强化学习三种类型。",
                category="人工智能",
                tags=["机器学习", "AI基础"]
            ),
            QAItem(
                question="深度学习与机器学习有什么区别？",
                answer="深度学习是机器学习的一个子领域，使用多层神经网络处理复杂问题。相比传统机器学习，深度学习能自动提取特征，但需要更多数据和计算资源。",
                category="人工智能",
                tags=["深度学习", "神经网络"]
            ),
            QAItem(
                question="如何使用Python进行数据分析？",
                answer="Python数据分析主要使用Pandas、NumPy、Matplotlib等库。基本流程包括：数据加载、数据清洗、数据探索、数据可视化和建模分析。",
                category="编程",
                tags=["Python", "数据分析", "Pandas"]
            ),
            QAItem(
                question="什么是区块链技术？",
                answer="区块链是一种分布式账本技术，通过密码学保证数据不可篡改。核心特点包括去中心化、透明性、不可篡改性，主要应用于加密货币、供应链、智能合约等。",
                category="技术",
                tags=["区块链", "分布式系统"]
            ),
            QAItem(
                question="如何开始机器学习项目？",
                answer="开始机器学习项目需要：1. 明确问题和目标 2. 收集和准备数据 3. 选择合适的算法 4. 训练模型 5. 评估和优化 6. 部署和监控",
                category="人工智能",
                tags=["机器学习", "项目实践"]
            )
        ]
        
        return sample_qa
    
    def record_user_feedback(self, qa_id: int, feedback_type: str, feedback_text: str = None):
        """记录用户反馈"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback (qa_id, feedback_type, feedback_text)
            VALUES (?, ?, ?)
        ''', (qa_id, feedback_type, feedback_text))
        
        conn.commit()
        conn.close()
        
        print(f"✅ 已记录反馈: {feedback_type}")
    
    def get_analytics(self) -> Dict:
        """获取系统分析数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本统计
        cursor.execute('SELECT COUNT(*) FROM qa_knowledge')
        total_qa = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversation_history')
        total_conversations = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM user_feedback')
        total_feedback = cursor.fetchone()[0]
        
        # 类别分布
        cursor.execute('SELECT category, COUNT(*) FROM qa_knowledge GROUP BY category')
        category_stats = dict(cursor.fetchall())
        
        # 热门问题
        cursor.execute('''
            SELECT question, usage_count FROM qa_knowledge 
            ORDER BY usage_count DESC LIMIT 5
        ''')
        popular_questions = cursor.fetchall()
        
        # 平均置信度
        cursor.execute('SELECT AVG(confidence) FROM qa_knowledge')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_qa': total_qa,
            'total_conversations': total_conversations,
            'total_feedback': total_feedback,
            'category_stats': category_stats,
            'popular_questions': popular_questions,
            'avg_confidence': avg_confidence
        }
    
    def demo_qa_system(self):
        """演示问答系统"""
        print("🚀 实战项目1：智能问答系统")
        print("=" * 60)
        
        # 加载示例知识
        sample_knowledge = self.load_sample_knowledge()
        
        print("📚 构建知识库...")
        for qa in sample_knowledge:
            self.add_qa_pair(qa.question, qa.answer, qa.category, qa.tags)
        
        # 演示问答功能
        test_questions = [
            "机器学习是什么？",
            "深度学习和机器学习有什么区别？",
            "Python怎么做数据分析？",
            "区块链能做什么？",
            "如何开始一个机器学习项目？"
        ]
        
        print("\n🎯 单轮问答演示")
        print("=" * 50)
        
        for question in test_questions:
            print(f"\n❓ 问题: {question}")
            answer = self.answer_question(question)
            print(f"🤖 回答: {answer['answer']}")
            print(f"📊 置信度: {answer['confidence']:.3f}")
        
        # 演示多轮对话
        print("\n🎯 多轮对话演示")
        print("=" * 50)
        
        session_id = "demo_session_001"
        conversation_context = []
        
        dialog_steps = [
            "我想学习机器学习",
            "从哪里开始比较好？",
            "需要准备什么数据？"
        ]
        
        for step in dialog_steps:
            print(f"\n❓ 用户: {step}")
            response = self.multi_turn_conversation(session_id, step, conversation_context)
            print(f"🤖 系统: {response['answer']}")
            
            conversation_context.append({
                'user': step,
                'system': response['answer']
            })
        
        # 显示分析数据
        print("\n📊 系统分析数据")
        print("=" * 50)
        
        analytics = self.get_analytics()
        print(f"📈 总问答对: {analytics['total_qa']}")
        print(f"💬 总对话数: {analytics['total_conversations']}")
        print(f"👍 总反馈数: {analytics['total_feedback']}")
        print(f"🎯 平均置信度: {analytics['avg_confidence']:.3f}")
        
        if analytics['category_stats']:
            print("\n📂 类别分布:")
            for category, count in analytics['category_stats'].items():
                print(f"   {category}: {count} 条")
        
        print("\n🎉 智能问答系统演示完成！")
        print("\n下一步：")
        print("   1. 运行项目: python main.py")
        print("   2. 启动Web服务: python app.py")
        print("   3. 测试API: python test_api.py")

def main():
    """主函数"""
    print("🚀 智能问答系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        qa_system = SmartQASystem()
        qa_system.demo_qa_system()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
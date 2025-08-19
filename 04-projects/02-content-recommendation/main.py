#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战项目2：内容推荐引擎
==================

基于用户行为和文本嵌入的个性化内容推荐系统。

项目功能：
1. 用户兴趣建模
2. 协同过滤推荐
3. 内容推荐算法
4. 实时推荐更新
5. A/B测试框架

技术栈：
- 用户行为分析
- 协同过滤算法
- 嵌入向量存储
- 实时计算
- 推荐解释
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict
import pickle
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

@dataclass
class User:
    """用户类"""
    user_id: str
    preferences: Dict[str, float]
    behavior_history: List[Dict]
    embedding: List[float] = None

@dataclass
class ContentItem:
    """内容类"""
    content_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    embedding: List[float] = None
    created_at: datetime = None

class ContentRecommendationSystem:
    """内容推荐引擎"""
    
    def __init__(self, db_path: str = "recommendation_system.db"):
        """初始化推荐系统"""
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        self.user_profiles = {}
        self.content_embeddings = {}
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 内容表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_items (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                tags TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 用户行为表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_id TEXT,
                behavior_type TEXT, -- 'view', 'like', 'share', 'skip'
                score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (content_id) REFERENCES content_items(id)
            )
        ''')
        
        # 推荐记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                content_id TEXT,
                recommendation_type TEXT, -- 'content', 'collaborative', 'hybrid'
                score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (content_id) REFERENCES content_items(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_content(self, content_id: str, title: str, content: str, category: str, tags: List[str]):
        """添加内容"""
        embedding = self.client.get_embedding(f"{title} {content}")
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO content_items (id, title, content, category, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (content_id, title, content, category, json.dumps(tags), embedding_str))
        
        conn.commit()
        conn.close()
        
        # 更新内存缓存
        self.content_embeddings[content_id] = {
            'title': title,
            'content': content,
            'category': category,
            'tags': tags,
            'embedding': embedding
        }
    
    def add_user(self, user_id: str, initial_preferences: Dict[str, float] = None):
        """添加用户"""
        preferences = initial_preferences or {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users (user_id, preferences)
            VALUES (?, ?)
        ''', (user_id, json.dumps(preferences)))
        
        conn.commit()
        conn.close()
        
        self.user_profiles[user_id] = {
            'preferences': preferences,
            'behavior_history': []
        }
    
    def record_user_behavior(self, user_id: str, content_id: str, behavior_type: str, score: float = 1.0):
        """记录用户行为"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_behaviors (user_id, content_id, behavior_type, score)
            VALUES (?, ?, ?, ?)
        ''', (user_id, content_id, behavior_type, score))
        
        conn.commit()
        conn.close()
        
        # 更新用户画像
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {'preferences': {}, 'behavior_history': []}
        
        self.user_profiles[user_id]['behavior_history'].append({
            'content_id': content_id,
            'behavior_type': behavior_type,
            'score': score,
            'timestamp': datetime.now()
        })
    
    def build_user_embedding(self, user_id: str) -> List[float]:
        """构建用户嵌入向量"""
        if user_id not in self.user_profiles:
            return [0.0] * 1024  # 默认嵌入
        
        # 获取用户行为历史
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT b.content_id, b.score, c.embedding 
            FROM user_behaviors b
            JOIN content_items c ON b.content_id = c.id
            WHERE b.user_id = ?
            ORDER BY b.timestamp DESC
            LIMIT 20
        ''', (user_id,))
        
        behaviors = cursor.fetchall()
        conn.close()
        
        if not behaviors:
            return [0.0] * 1024
        
        # 计算加权平均嵌入
        weighted_embeddings = []
        total_weight = 0
        
        for content_id, score, embedding_str in behaviors:
            embedding = json.loads(embedding_str)
            weight = score
            weighted_embeddings.append([e * weight for e in embedding])
            total_weight += weight
        
        if total_weight > 0:
            user_embedding = [sum(col) / total_weight for col in zip(*weighted_embeddings)]
        else:
            user_embedding = [0.0] * 1024
        
        # 保存用户嵌入
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET embedding = ? WHERE user_id = ?
        ''', (json.dumps(user_embedding), user_id))
        conn.commit()
        conn.close()
        
        return user_embedding
    
    def content_based_recommendation(self, user_id: str, limit: int = 10) -> List[Dict]:
        """基于内容的推荐"""
        user_embedding = self.build_user_embedding(user_id)
        
        # 获取所有内容
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, content, category, tags, embedding
            FROM content_items
        ''')
        contents = cursor.fetchall()
        conn.close()
        
        # 计算相似度
        recommendations = []
        user_vec = np.array(user_embedding)
        
        for content_id, title, content, category, tags_str, embedding_str in contents:
            # 检查用户是否已经交互过
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM user_behaviors
                WHERE user_id = ? AND content_id = ?
            ''', (user_id, content_id))
            
            if cursor.fetchone()[0] > 0:  # 用户已交互过
                conn.close()
                continue
            conn.close()
            
            content_vec = np.array(json.loads(embedding_str))
            similarity = np.dot(user_vec, content_vec) / (
                np.linalg.norm(user_vec) * np.linalg.norm(content_vec)
            )
            
            recommendations.append({
                'content_id': content_id,
                'title': title,
                'content': content,
                'category': category,
                'tags': json.loads(tags_str),
                'score': similarity,
                'type': 'content_based'
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:limit]
    
    def collaborative_filtering(self, user_id: str, limit: int = 10) -> List[Dict]:
        """协同过滤推荐"""
        # 获取所有用户行为
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, content_id, score
            FROM user_behaviors
            WHERE behavior_type IN ('like', 'view')
        ''')
        behaviors = cursor.fetchall()
        conn.close()
        
        # 构建用户-内容矩阵
        user_content_matrix = defaultdict(dict)
        for user, content, score in behaviors:
            user_content_matrix[user][content] = score
        
        # 找到相似用户
        if user_id not in user_content_matrix:
            return []
        
        target_user_items = user_content_matrix[user_id]
        similar_users = []
        
        for other_user, items in user_content_matrix.items():
            if other_user == user_id:
                continue
            
            # 计算用户相似度
            common_items = set(target_user_items.keys()) & set(items.keys())
            if len(common_items) > 0:
                similarities = []
                for item in common_items:
                    similarities.append(
                        (target_user_items[item] - items[item]) ** 2
                    )
                
                similarity = 1 / (1 + np.sqrt(sum(similarities) / len(similarities)))
                similar_users.append((other_user, similarity))
        
        # 按相似度排序
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        # 推荐内容
        recommendations = []
        for similar_user, similarity in similar_users[:5]:
            for content_id, score in user_content_matrix[similar_user].items():
                if content_id not in target_user_items:
                    recommendations.append({
                        'content_id': content_id,
                        'score': score * similarity,
                        'type': 'collaborative',
                        'similar_user': similar_user
                    })
        
        # 获取内容详情
        final_recommendations = []
        for rec in recommendations[:limit]:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, content, category, tags
                FROM content_items WHERE id = ?
            ''', (rec['content_id'],))
            
            content = cursor.fetchone()
            conn.close()
            
            if content:
                final_recommendations.append({
                    'content_id': content[0],
                    'title': content[1],
                    'content': content[2],
                    'category': content[3],
                    'tags': json.loads(content[4]),
                    'score': rec['score'],
                    'type': rec['type']
                })
        
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return final_recommendations[:limit]
    
    def hybrid_recommendation(self, user_id: str, limit: int = 10) -> List[Dict]:
        """混合推荐"""
        # 获取基于内容的推荐
        content_recs = self.content_based_recommendation(user_id, limit * 2)
        
        # 获取协同过滤推荐
        collab_recs = self.collaborative_filtering(user_id, limit * 2)
        
        # 合并和去重
        all_recommendations = {}
        
        # 添加基于内容的推荐
        for rec in content_recs:
            rec_id = rec['content_id']
            if rec_id not in all_recommendations:
                all_recommendations[rec_id] = rec
                all_recommendations[rec_id]['weight'] = rec['score'] * 0.6  # 内容权重
        
        # 添加协同过滤推荐
        for rec in collab_recs:
            rec_id = rec['content_id']
            if rec_id in all_recommendations:
                all_recommendations[rec_id]['weight'] += rec['score'] * 0.4
            else:
                all_recommendations[rec_id] = rec
                all_recommendations[rec_id]['weight'] = rec['score'] * 0.4
        
        # 排序并返回
        recommendations = list(all_recommendations.values())
        recommendations.sort(key=lambda x: x['weight'], reverse=True)
        
        # 记录推荐
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for rec in recommendations[:limit]:
            cursor.execute('''
                INSERT INTO recommendations (user_id, content_id, recommendation_type, score)
                VALUES (?, ?, ?, ?)
            ''', (user_id, rec['content_id'], rec['type'], rec['weight']))
        conn.commit()
        conn.close()
        
        return recommendations[:limit]
    
    def load_sample_data(self):
        """加载示例数据"""
        # 示例内容
        sample_contents = [
            {
                'content_id': 'tech_001',
                'title': '人工智能入门指南',
                'content': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的机器。',
                'category': '技术',
                'tags': ['AI', '入门', '机器学习']
            },
            {
                'content_id': 'tech_002',
                'title': '深度学习实战',
                'content': '深度学习使用多层神经网络处理复杂问题，在图像识别、自然语言处理等领域取得突破。',
                'category': '技术',
                'tags': ['深度学习', '神经网络', '实战']
            },
            {
                'content_id': 'sports_001',
                'title': '2024奥运会精彩回顾',
                'content': '2024年巴黎奥运会圆满落幕，中国代表团表现出色，获得多枚金牌。',
                'category': '体育',
                'tags': ['奥运会', '体育', '中国']
            },
            {
                'content_id': 'finance_001',
                'title': '投资理财入门',
                'content': '投资理财是实现财务自由的重要途径，需要学习基本的理财知识和风险控制。',
                'category': '财经',
                'tags': ['理财', '投资', '入门']
            }
        ]
        
        # 添加内容
        for content in sample_contents:
            self.add_content(
                content['content_id'],
                content['title'],
                content['content'],
                content['category'],
                content['tags']
            )
        
        # 添加用户
        self.add_user('user_001', {'技术': 0.8, '体育': 0.3, '财经': 0.5})
        self.add_user('user_002', {'体育': 0.9, '技术': 0.4, '财经': 0.2})
        
        # 添加用户行为
        behaviors = [
            ('user_001', 'tech_001', 'like', 1.0),
            ('user_001', 'tech_002', 'view', 0.8),
            ('user_002', 'sports_001', 'like', 1.0),
            ('user_001', 'finance_001', 'view', 0.6)
        ]
        
        for user_id, content_id, behavior, score in behaviors:
            self.record_user_behavior(user_id, content_id, behavior, score)
    
    def demo_recommendation_system(self):
        """演示推荐系统"""
        print("🚀 实战项目2：内容推荐引擎")
        print("=" * 60)
        
        # 加载示例数据
        print("📊 加载示例数据...")
        self.load_sample_data()
        
        # 演示推荐功能
        test_users = ['user_001', 'user_002']
        
        for user_id in test_users:
            print(f"\n👤 为用户 {user_id} 生成推荐")
            print("=" * 50)
            
            # 基于内容的推荐
            content_recs = self.content_based_recommendation(user_id, limit=3)
            print("\n🎯 基于内容的推荐:")
            for rec in content_recs:
                print(f"   📖 {rec['title']} (分数: {rec['score']:.3f})")
            
            # 协同过滤推荐
            collab_recs = self.collaborative_filtering(user_id, limit=3)
            print("\n🎯 协同过滤推荐:")
            for rec in collab_recs:
                print(f"   📖 {rec['title']} (分数: {rec['score']:.3f})")
            
            # 混合推荐
            hybrid_recs = self.hybrid_recommendation(user_id, limit=3)
            print("\n🎯 混合推荐:")
            for rec in hybrid_recs:
                print(f"   📖 {rec['title']} (分数: {rec['weight']:.3f}, 类型: {rec['type']})")
        
        print("\n🎉 内容推荐引擎演示完成！")
        print("\n下一步：")
        print("   1. 运行实时推荐服务: python realtime_service.py")
        print("   2. 启动Web界面: python web_app.py")
        print("   3. 测试推荐API: python test_api.py")

def main():
    """主函数"""
    print("🚀 内容推荐引擎")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        recommender = ContentRecommendationSystem()
        recommender.demo_recommendation_system()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
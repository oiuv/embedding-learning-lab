#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级项目1：实时推荐系统
==================

构建基于实时用户行为的动态推荐系统。

项目目标：
1. 实时行为捕获
2. 增量模型更新
3. 冷启动问题解决
4. A/B测试框架
5. 实时性能监控
6. 推荐解释系统
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime, timedelta
import threading
import time
import queue
from dataclasses import dataclass
import random

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

@dataclass
class UserAction:
    """用户行为事件"""
    user_id: str
    item_id: str
    action_type: str  # 'view', 'like', 'purchase', 'skip'
    timestamp: datetime
    context: Dict

@dataclass
class Recommendation:
    """推荐结果"""
    user_id: str
    item_id: str
    score: float
    reason: str
    timestamp: datetime
    is_real_time: bool = False

class RealTimeRecommendationSystem:
    """实时推荐系统"""
    
    def __init__(self, db_path: str = "realtime_recommendation.db"):
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        
        # 实时处理队列
        self.action_queue = queue.Queue()
        self.recommendation_cache = {}
        self.online_users = set()
        
        # 模型参数
        self.learning_rate = 0.01
        self.regularization = 0.001
        self.latent_dim = 50
        
        # 性能监控
        self.metrics = {
            'total_recommendations': 0,
            'total_clicks': 0,
            'average_response_time': 0,
            'cache_hit_rate': 0
        }
        
    def init_database(self):
        """初始化实时数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 用户行为表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                score REAL DEFAULT 1.0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                context TEXT
            )
        ''')
        
        # 实时推荐表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                score REAL NOT NULL,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                real_time BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # A/B测试结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                test_group TEXT NOT NULL,
                recommendation_id TEXT NOT NULL,
                clicked BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 性能监控表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_background_worker(self):
        """启动后台处理线程"""
        self.worker_thread = threading.Thread(target=self._process_actions, daemon=True)
        self.worker_thread.start()
        
        self.metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.metrics_thread.start()
    
    def _process_actions(self):
        """后台处理用户行为"""
        while True:
            try:
                action = self.action_queue.get(timeout=1)
                self._update_user_model(action)
                self._update_recommendations(action.user_id)
                self.action_queue.task_done()
            except queue.Empty:
                continue
    
    def _collect_metrics(self):
        """收集性能指标"""
        while True:
            time.sleep(10)  # 每10秒收集一次
            self._record_metric('queue_size', self.action_queue.qsize())
            self._record_metric('active_users', len(self.online_users))
    
    def record_user_action(self, action: UserAction):
        """记录用户行为到队列"""
        self.action_queue.put(action)
        
        # 记录到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO user_actions (user_id, item_id, action_type, score, context)
            VALUES (?, ?, ?, ?, ?)
        ''', (action.user_id, action.item_id, action.action_type, 
              self._get_action_score(action.action_type), json.dumps(action.context)))
        conn.commit()
        conn.close()
        
        # 更新在线用户
        self.online_users.add(action.user_id)
    
    def _get_action_score(self, action_type: str) -> float:
        """获取行为分数"""
        scores = {
            'view': 1.0,
            'like': 3.0,
            'purchase': 5.0,
            'skip': -1.0
        }
        return scores.get(action_type, 1.0)
    
    def _update_user_model(self, action: UserAction):
        """更新用户模型（增量更新）"""
        user_id = action.user_id
        item_id = action.item_id
        score = self._get_action_score(action.action_type)
        
        # 简化的增量矩阵分解
        if user_id not in self.user_vectors:
            self.user_vectors[user_id] = np.random.normal(0, 0.1, self.latent_dim)
        
        if item_id not in self.item_vectors:
            self.item_vectors[item_id] = np.random.normal(0, 0.1, self.latent_dim)
        
        # 梯度下降更新
        user_vec = self.user_vectors[user_id]
        item_vec = self.item_vectors[item_id]
        
        prediction = np.dot(user_vec, item_vec)
        error = score - prediction
        
        # 更新向量
        user_vec += self.learning_rate * (error * item_vec - self.regularization * user_vec)
        item_vec += self.learning_rate * (error * user_vec - self.regularization * item_vec)
        
        self.user_vectors[user_id] = user_vec
        self.item_vectors[item_id] = item_vec
    
    def _update_recommendations(self, user_id: str):
        """更新用户推荐"""
        if user_id not in self.user_vectors:
            self._handle_cold_start(user_id)
            return
        
        user_vec = self.user_vectors[user_id]
        
        # 计算所有物品的分数
        recommendations = []
        for item_id, item_vec in self.item_vectors.items():
            score = np.dot(user_vec, item_vec)
            
            # 添加多样性因子
            diversity_score = self._calculate_diversity(user_id, item_id)
            final_score = score + 0.1 * diversity_score
            
            recommendations.append(Recommendation(
                user_id=user_id,
                item_id=item_id,
                score=final_score,
                reason=self._generate_explanation(user_id, item_id),
                timestamp=datetime.now(),
                is_real_time=True
            ))
        
        # 排序并缓存
        recommendations.sort(key=lambda x: x.score, reverse=True)
        self.recommendation_cache[user_id] = recommendations[:10]
        
        # 记录到数据库
        self._save_recommendations_to_db(user_id, recommendations[:10])
    
    def _save_recommendations_to_db(self, user_id: str, recommendations: List[Recommendation]):
        """保存推荐到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for rec in recommendations:
            cursor.execute('''
                INSERT INTO realtime_recommendations (user_id, item_id, score, reason, real_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (rec.user_id, rec.item_id, rec.score, rec.reason, rec.is_real_time))
        
        conn.commit()
        conn.close()
    
    def _handle_cold_start(self, user_id: str):
        """处理冷启动问题"""
        # 使用流行度推荐
        popular_items = self._get_popular_items()
        
        recommendations = []
        for item_id, popularity in popular_items:
            recommendations.append(Recommendation(
                user_id=user_id,
                item_id=item_id,
                score=popularity,
                reason="基于热门推荐",
                timestamp=datetime.now(),
                is_real_time=False
            ))
        
        self.recommendation_cache[user_id] = recommendations
    
    def _get_popular_items(self, limit: int = 10) -> List[Tuple[str, float]]:
        """获取热门物品"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT item_id, AVG(score) as avg_score, COUNT(*) as action_count
            FROM user_actions
            WHERE action_type IN ('view', 'like', 'purchase')
            GROUP BY item_id
            ORDER BY avg_score DESC, action_count DESC
            LIMIT ?
        ''', (limit,))
        
        items = cursor.fetchall()
        conn.close()
        
        return [(item[0], item[1]) for item in items]
    
    def _calculate_diversity(self, user_id: str, item_id: str) -> float:
        """计算多样性分数"""
        if user_id not in self.recommendation_cache:
            return 0.0
        
        # 避免推荐过于相似的物品
        user_recommendations = self.recommendation_cache[user_id]
        item_vec = self.item_vectors[item_id]
        
        similarities = []
        for rec in user_recommendations[:5]:  # 检查前5个推荐
            if rec.item_id in self.item_vectors:
                sim = np.dot(item_vec, self.item_vectors[rec.item_id])
                similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity  # 奖励不相似的物品
        
        return 0.0
    
    def _generate_explanation(self, user_id: str, item_id: str) -> str:
        """生成推荐解释"""
        # 基于用户行为的解释
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT action_type, COUNT(*) as count
            FROM user_actions
            WHERE user_id = ?
            GROUP BY action_type
            ORDER BY count DESC
        ''', (user_id,))
        
        behaviors = cursor.fetchall()
        conn.close()
        
        if behaviors:
            top_behavior = behaviors[0][0]
            return f"基于您对{top_behavior}类物品的偏好"
        
        return "为您推荐的新物品"
    
    def get_realtime_recommendations(self, user_id: str, limit: int = 10) -> List[Recommendation]:
        """获取实时推荐"""
        start_time = time.time()
        
        # 检查缓存
        if user_id in self.recommendation_cache:
            recommendations = self.recommendation_cache[user_id][:limit]
            cache_hit = True
        else:
            # 冷启动处理
            self._handle_cold_start(user_id)
            recommendations = self.recommendation_cache[user_id][:limit]
            cache_hit = False
        
        response_time = time.time() - start_time
        
        # 更新指标
        self.metrics['total_recommendations'] += 1
        if cache_hit:
            self.metrics['cache_hit_rate'] = (
                (self.metrics['cache_hit_rate'] * (self.metrics['total_recommendations'] - 1) + 1) /
                self.metrics['total_recommendations']
            )
        
        self._record_metric('response_time', response_time)
        
        return recommendations
    
    def ab_test(self, user_id: str, test_group: str = 'A') -> List[Recommendation]:
        """A/B测试"""
        if test_group == 'A':
            # 控制组：现有算法
            return self.get_realtime_recommendations(user_id)
        elif test_group == 'B':
            # 实验组：新算法（这里简化为热门物品）
            popular_items = self._get_popular_items()
            recommendations = []
            
            for item_id, score in popular_items:
                recommendations.append(Recommendation(
                    user_id=user_id,
                    item_id=item_id,
                    score=score * 1.2,  # 实验组分数提升
                    reason="A/B测试实验组",
                    timestamp=datetime.now()
                ))
            
            return recommendations[:10]
    
    def record_ab_test_result(self, user_id: str, test_group: str, item_id: str, clicked: bool):
        """记录A/B测试结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_test_results (user_id, test_group, recommendation_id, clicked)
            VALUES (?, ?, ?, ?)
        ''', (user_id, test_group, item_id, clicked))
        
        conn.commit()
        conn.close()
    
    def _record_metric(self, metric_name: str, value: float):
        """记录性能指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics (metric_name, value)
            VALUES (?, ?)
        ''', (metric_name, value))
        
        conn.commit()
        conn.close()
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # A/B测试结果
        cursor.execute('''
            SELECT test_group, COUNT(*), SUM(CASE WHEN clicked THEN 1 ELSE 0 END)
            FROM ab_test_results
            GROUP BY test_group
        ''')
        
        ab_results = cursor.fetchall()
        
        # 性能指标
        cursor.execute('''
            SELECT metric_name, AVG(value), MAX(value), MIN(value)
            FROM performance_metrics
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY metric_name
        ''')
        
        metrics = cursor.fetchall()
        
        conn.close()
        
        return {
            'ab_test': {
                'group_A': {'total': ab_results[0][1], 'clicked': ab_results[0][2]} if ab_results else {'total': 0, 'clicked': 0},
                'group_B': {'total': ab_results[1][1], 'clicked': ab_results[1][2]} if len(ab_results) > 1 else {'total': 0, 'clicked': 0}
            },
            'metrics': {row[0]: {'avg': row[1], 'max': row[2], 'min': row[3]} for row in metrics},
            'current_cache_size': len(self.recommendation_cache),
            'active_users': len(self.online_users)
        }
    
    def simulate_user_behavior(self, user_id: str, item_id: str, action: str):
        """模拟用户行为"""
        user_action = UserAction(
            user_id=user_id,
            item_id=item_id,
            action_type=action,
            timestamp=datetime.now(),
            context={'source': 'simulation'}
        )
        
        self.record_user_action(user_action)
    
    def load_sample_data(self):
        """加载示例数据"""
        # 示例物品
        sample_items = [
            ('item_001', '机器学习入门', 'AI基础课程'),
            ('item_002', '深度学习进阶', '高级AI技术'),
            ('item_003', 'Python数据分析', '数据科学工具'),
            ('item_004', 'Web开发实战', '前端技术'),
            ('item_005', '移动应用开发', '移动技术'),
            ('item_006', '云计算基础', '云服务'),
            ('item_007', '区块链原理', '分布式技术'),
            ('item_008', '网络安全', '安全技术')
        ]
        
        # 示例用户行为
        sample_behaviors = [
            ('user_001', 'item_001', 'view'),
            ('user_001', 'item_002', 'like'),
            ('user_001', 'item_003', 'purchase'),
            ('user_002', 'item_004', 'view'),
            ('user_002', 'item_005', 'like'),
            ('user_003', 'item_006', 'view'),
            ('user_003', 'item_007', 'like'),
            ('user_003', 'item_008', 'purchase')
        ]
        
        return sample_items, sample_behaviors
    
    def demo_realtime_system(self):
        """演示实时推荐系统"""
        print("⚡ 高级项目：实时推荐系统")
        print("=" * 60)
        
        # 初始化系统
        self.user_vectors = {}
        self.item_vectors = {}
        self.start_background_worker()
        
        # 加载数据
        items, behaviors = self.load_sample_data()
        
        print("📊 初始化系统...")
        print(f"   物品数量: {len(items)}")
        print(f"   用户行为: {len(behaviors)}")
        
        # 模拟用户行为
        print("\n⚡ 模拟实时用户行为")
        print("-" * 30)
        
        for user_id, item_id, action in behaviors:
            print(f"   用户 {user_id} {action} 物品 {item_id}")
            self.simulate_user_behavior(user_id, item_id, action)
            
            # 获取实时推荐
            if user_id in self.recommendation_cache:
                recommendations = self.get_realtime_recommendations(user_id, limit=3)
                print(f"      推荐: {[rec.item_id for rec in recommendations]}")
            
            time.sleep(0.5)  # 模拟实时延迟
        
        # A/B测试
        print("\n🧪 A/B测试演示")
        print("-" * 30)
        
        test_user = 'user_test'
        
        # 控制组
        recs_a = self.ab_test(test_user, 'A')
        print(f"   控制组推荐: {[rec.item_id for rec in recs_a[:3]]}")
        
        # 实验组
        recs_b = self.ab_test(test_user, 'B')
        print(f"   实验组推荐: {[rec.item_id for rec in recs_b[:3]]}")
        
        # 模拟点击
        if recs_a:
            self.record_ab_test_result(test_user, 'A', recs_a[0].item_id, clicked=True)
        if recs_b:
            self.record_ab_test_result(test_user, 'B', recs_b[0].item_id, clicked=False)
        
        # 性能报告
        print("\n📊 性能报告")
        print("-" * 30)
        
        report = self.get_performance_report()
        
        print(f"   活跃用户数: {report['active_users']}")
        print(f"   缓存大小: {report['current_cache_size']}")
        
        if 'metrics' in report:
            for metric, values in report['metrics'].items():
                print(f"   {metric}: avg={values['avg']:.3f}s")
        
        print("\n✅ 实时推荐系统演示完成！")
        print("\n🎓 关键技术：")
        print("   • 增量学习：实时更新用户模型")
        print("   • 冷启动：新用户的推荐策略")
        print("   • A/B测试：算法效果对比")
        print("   • 性能监控：系统运行状态")

def main():
    """主函数"""
    print("⚡ 高级挑战：实时推荐系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        system = RealTimeRecommendationSystem()
        system.demo_realtime_system()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级功能第2课：异常检测系统
=========================

基于文本嵌入的异常检测系统，用于识别垃圾内容、欺诈信息等。

学习目标：
1. 理解异常检测原理
2. 实现基于嵌入的异常识别
3. 垃圾内容过滤系统
4. 欺诈检测算法
5. 实时监控系统
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

# 设置中文字体支持
from matplotlib import rcParams
rcParams['font.family'] = ['sans-serif']
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class AnomalyDetectionSystem:
    """异常检测系统"""
    
    def __init__(self):
        """初始化异常检测系统"""
        self.client = EmbeddingClient()
        self.normal_profiles = {}
        self.thresholds = {}
        self.detection_models = {}
        
    def load_sample_data(self) -> Dict[str, List[str]]:
        """加载示例数据（正常和异常文本）"""
        sample_data = {
            "正常评论": [
                "这个产品质量很好，物流速度快，包装完好",
                "客服态度很好，解决问题及时，很满意这次购物体验",
                "价格合理，物有所值，会再次购买推荐给大家",
                "商品描述准确，和图片一致，没有色差",
                "发货迅速，物流跟踪信息准确，收货及时"
            ],
            "垃圾广告": [
                "🔥🔥🔥限时抢购！点击链接获取优惠🔥🔥🔥",
                "加我微信：XXXXX，获取更多优惠信息",
                "特价商品，数量有限，速抢！联系QQ：123456",
                "🔥超值优惠🔥不要错过机会🔥立即购买🔥",
                "专业刷单团队，安全可靠，请联系客服"
            ],
            "虚假评论": [
                "这是一个很好的产品，我从来没有用过这么好的东西",
                "绝对完美，没有任何缺点，100%推荐购买",
                "太棒了太棒了太棒了重要的事情说三遍",
                "这个产品改变了我的生活，强烈推荐给大家",
                "我从来没有见过这么好的商品，必须给五星"
            ],
            "恶意攻击": [
                "垃圾产品，千万别买，骗子商家",
                "质量差到极点，完全是虚假宣传",
                "客服态度恶劣，问题不解决还骂人",
                "收到货就坏了，商家不处理还推卸责任",
                "浪费钱，后悔购买，大家千万别上当"
            ],
            "正常技术讨论": [
                "这个算法的实现思路很清晰，代码质量不错",
                "文档写得很详细，对理解项目很有帮助",
                "性能测试结果符合预期，建议继续优化",
                "API设计合理，接口调用简单方便",
                "测试覆盖率很高，代码质量有保障"
            ]
        }
        return sample_data
    
    def create_normal_profile(self, normal_texts: List[str], domain: str = "default") -> Dict:
        """创建正常文本特征档案"""
        print(f"🎯 创建{domain}领域正常文本档案...")
        
        embeddings = []
        for text in normal_texts:
            embedding = self.client.get_embedding(text)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        
        # 计算正常文本的中心向量
        center_vector = np.mean(embeddings_array, axis=0)
        
        # 计算每个文本与中心的距离作为阈值基础
        distances = []
        for emb in embeddings_array:
            distance = np.linalg.norm(emb - center_vector)
            distances.append(distance)
        
        # 设置异常阈值（均值 + 2标准差）
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + 2 * std_distance
        
        self.normal_profiles[domain] = {
            'center_vector': center_vector,
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'threshold': threshold,
            'normal_count': len(normal_texts)
        }
        
        print(f"   ✅ 阈值设定: {threshold:.3f}")
        return self.normal_profiles[domain]
    
    def distance_based_detection(self, texts: List[str], domain: str = "default") -> List[Dict]:
        """基于距离的异常检测"""
        if domain not in self.normal_profiles:
            raise ValueError(f"请先为{domain}领域创建正常档案")
        
        profile = self.normal_profiles[domain]
        center_vector = profile['center_vector']
        threshold = profile['threshold']
        
        results = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            distance = np.linalg.norm(np.array(embedding) - center_vector)
            
            is_anomaly = distance > threshold
            confidence = min(distance / threshold, 1.0) if is_anomaly else 1.0 - distance / threshold
            
            results.append({
                'text': text,
                'distance': distance,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'method': 'distance_based'
            })
        
        return results
    
    def isolation_forest_detection(self, texts: List[str]) -> List[Dict]:
        """Isolation Forest异常检测"""
        print("🎯 Isolation Forest异常检测...")
        
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 使用Isolation Forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = isolation_forest.fit_predict(embeddings_array)
        scores = isolation_forest.decision_function(embeddings_array)
        
        results = []
        for i, (text, pred, score) in enumerate(zip(texts, predictions, scores)):
            is_anomaly = pred == -1  # -1表示异常
            confidence = abs(score)  # 分数绝对值作为置信度
            
            results.append({
                'text': text,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'score': score,
                'method': 'isolation_forest'
            })
        
        return results
    
    def lof_detection(self, texts: List[str]) -> List[Dict]:
        """Local Outlier Factor异常检测"""
        print("🎯 LOF异常检测...")
        
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 使用LOF
        lof = LocalOutlierFactor(n_neighbors=5, contamination=0.1)
        predictions = lof.fit_predict(embeddings_array)
        scores = lof.negative_outlier_factor_
        
        results = []
        for i, (text, pred, score) in enumerate(zip(texts, predictions, scores)):
            is_anomaly = pred == -1
            confidence = abs(score)
            
            results.append({
                'text': text,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'score': score,
                'method': 'lof'
            })
        
        return results
    
    def one_class_svm_detection(self, texts: List[str], normal_texts: List[str]) -> List[Dict]:
        """One-Class SVM异常检测"""
        print("🎯 One-Class SVM异常检测...")
        
        # 使用正常文本训练模型
        normal_embeddings = [self.client.get_embedding(text) for text in normal_texts]
        normal_array = np.array(normal_embeddings)
        
        # 训练One-Class SVM
        svm = OneClassSVM(kernel='rbf', nu=0.1)
        svm.fit(normal_array)
        
        # 检测异常
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        predictions = svm.predict(embeddings_array)
        scores = svm.decision_function(embeddings_array)
        
        results = []
        for i, (text, pred, score) in enumerate(zip(texts, predictions, scores)):
            is_anomaly = pred == -1
            confidence = abs(score)
            
            results.append({
                'text': text,
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'score': score,
                'method': 'one_class_svm'
            })
        
        return results
    
    def ensemble_detection(self, texts: List[str], normal_texts: List[str], domain: str = "default") -> List[Dict]:
        """集成异常检测"""
        print("🎯 集成异常检测...")
        
        # 获取所有检测方法的结果
        distance_results = self.distance_based_detection(texts, domain)
        isolation_results = self.isolation_forest_detection(texts)
        lof_results = self.lof_detection(texts)
        svm_results = self.one_class_svm_detection(texts, normal_texts)
        
        # 集成结果
        ensemble_results = []
        for i, text in enumerate(texts):
            # 收集所有方法的检测结果
            votes = []
            confidences = []
            
            for method_results in [distance_results, isolation_results, lof_results, svm_results]:
                vote = 1 if method_results[i]['is_anomaly'] else 0
                confidence = method_results[i]['confidence']
                votes.append(vote)
                confidences.append(confidence)
            
            # 投票结果
            ensemble_vote = sum(votes) / len(votes)
            ensemble_confidence = np.mean(confidences)
            
            # 多数投票决定最终结果
            is_anomaly = ensemble_vote > 0.5
            
            ensemble_results.append({
                'text': text,
                'is_anomaly': is_anomaly,
                'confidence': ensemble_confidence,
                'vote_score': ensemble_vote,
                'individual_results': {
                    'distance': distance_results[i],
                    'isolation_forest': isolation_results[i],
                    'lof': lof_results[i],
                    'svm': svm_results[i]
                },
                'method': 'ensemble'
            })
        
        return ensemble_results
    
    def visualize_anomaly_detection(self, texts: List[str], results: List[Dict], method_name: str):
        """可视化异常检测结果"""
        embeddings = [self.client.get_embedding(text) for text in texts]
        embeddings_array = np.array(embeddings)
        
        # 降维可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_array)
        
        plt.figure(figsize=(12, 8))
        
        # 分离正常和异常点
        normal_mask = [not r['is_anomaly'] for r in results]
        anomaly_mask = [r['is_anomaly'] for r in results]
        
        plt.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1], 
                   c='blue', alpha=0.6, label='正常文本', s=100)
        plt.scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1], 
                   c='red', alpha=0.8, label='异常文本', s=100, marker='^')
        
        # 添加置信度标签
        for i, (x, y) in enumerate(embeddings_2d):
            if results[i]['is_anomaly']:
                plt.annotate(f"{results[i]['confidence']:.2f}", 
                           (x, y), fontsize=8, color='red')
        
        plt.title(f"{method_name}异常检测结果可视化", fontsize=14)
        plt.xlabel("主成分1", fontsize=12)
        plt.ylabel("主成分2", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"03-advanced/anomaly_{method_name.lower().replace(' ', '_')}_visualization.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 可视化已保存: {filename}")
    
    def real_time_monitoring(self, text_stream: List[str], normal_profile: Dict) -> List[Dict]:
        """实时异常监控"""
        print("🎯 实时异常监控...")
        
        alerts = []
        for i, text in enumerate(text_stream):
            # 快速检测
            distance_results = self.distance_based_detection([text], "default")
            
            if distance_results[0]['is_anomaly']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'text': text,
                    'anomaly_type': 'real_time_detection',
                    'confidence': distance_results[0]['confidence'],
                    'severity': 'high' if distance_results[0]['confidence'] > 0.8 else 'medium'
                })
        
        return alerts
    
    def generate_detection_report(self, texts: List[str], results: List[Dict]) -> Dict:
        """生成异常检测报告"""
        total_texts = len(texts)
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        normal_count = total_texts - anomaly_count
        
        # 按置信度分类
        high_confidence = [r for r in results if r['is_anomaly'] and r['confidence'] > 0.8]
        medium_confidence = [r for r in results if r['is_anomaly'] and 0.5 <= r['confidence'] <= 0.8]
        low_confidence = [r for r in results if r['is_anomaly'] and r['confidence'] < 0.5]
        
        report = {
            'total_texts': total_texts,
            'anomaly_count': anomaly_count,
            'normal_count': normal_count,
            'anomaly_rate': anomaly_count / total_texts if total_texts > 0 else 0,
            'high_confidence_anomalies': len(high_confidence),
            'medium_confidence_anomalies': len(medium_confidence),
            'low_confidence_anomalies': len(low_confidence),
            'anomaly_texts': [r['text'] for r in results if r['is_anomaly']],
            'method': results[0]['method'] if results else 'unknown'
        }
        
        return report
    
    def demo_anomaly_detection(self):
        """演示异常检测系统"""
        print("🚀 高级功能第2课：异常检测系统")
        print("=" * 60)
        
        # 加载示例数据
        sample_data = self.load_sample_data()
        
        print("📊 数据准备...")
        normal_texts = sample_data["正常评论"] + sample_data["正常技术讨论"]
        test_texts = (
            sample_data["正常评论"][:2] +
            sample_data["垃圾广告"] +
            sample_data["虚假评论"] +
            sample_data["恶意攻击"]
        )
        
        # 创建正常档案
        profile = self.create_normal_profile(normal_texts, "评论系统")
        
        print("\n🔍 第1部分：基于距离的异常检测")
        print("=" * 50)
        
        distance_results = self.distance_based_detection(test_texts, "评论系统")
        self._print_detection_results(distance_results, "距离法")
        
        print("\n🔍 第2部分：Isolation Forest异常检测")
        print("=" * 50)
        
        isolation_results = self.isolation_forest_detection(test_texts)
        self._print_detection_results(isolation_results, "Isolation Forest")
        
        print("\n🔍 第3部分：Local Outlier Factor检测")
        print("=" * 50)
        
        lof_results = self.lof_detection(test_texts)
        self._print_detection_results(lof_results, "LOF")
        
        print("\n🔍 第4部分：One-Class SVM检测")
        print("=" * 50)
        
        svm_results = self.one_class_svm_detection(test_texts, normal_texts)
        self._print_detection_results(svm_results, "One-Class SVM")
        
        print("\n🔍 第5部分：集成异常检测")
        print("=" * 50)
        
        ensemble_results = self.ensemble_detection(test_texts, normal_texts, "评论系统")
        self._print_detection_results(ensemble_results, "集成方法")
        
        # 生成检测报告
        report = self.generate_detection_report(test_texts, ensemble_results)
        print(f"\n📊 检测统计报告:")
        print(f"   总文本数: {report['total_texts']}")
        print(f"   异常数: {report['anomaly_count']}")
        print(f"   异常率: {report['anomaly_rate']:.2%}")
        print(f"   高置信度异常: {report['high_confidence_anomalies']}")
        print(f"   中置信度异常: {report['medium_confidence_anomalies']}")
        print(f"   低置信度异常: {report['low_confidence_anomalies']}")
        
        # 可视化结果
        self.visualize_anomaly_detection(test_texts, ensemble_results, "Ensemble")
    
    def _print_detection_results(self, results: List[Dict], method_name: str):
        """打印检测结果"""
        anomalies = [r for r in results if r['is_anomaly']]
        
        print(f"\n{method_name}检测结果:")
        print(f"   检测到 {len(anomalies)} 个异常文本")
        
        for anomaly in anomalies[:3]:  # 显示前3个
            print(f"   ⚠️ {anomaly['text'][:50]}... (置信度: {anomaly['confidence']:.3f})")

def main():
    """主函数"""
    print("🚀 异常检测系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        detector = AnomalyDetectionSystem()
        detector.demo_anomaly_detection()
        
        print("\n🎉 异常检测演示完成！")
        print("\n核心技术总结：")
        print("   • 多种异常检测算法")
        print("   • 集成检测方法")
        print("   • 实时监控系统")
        print("   • 可视化分析")
        print("\n实际应用场景：")
        print("   • 垃圾内容过滤")
        print("   • 欺诈检测")
        print("   • 内容质量监控")
        print("   • 实时风控系统")
        print("\n下一课：03-03-visualization.py - 高级可视化")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
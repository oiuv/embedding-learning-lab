#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级课程第2课：文本分类系统
=======================

基于文本嵌入的文本自动分类系统实现。
通过向量化技术实现新闻、评论、邮件等文本的自动分类。

学习目标：
1. 理解文本分类的工作原理
2. 掌握基于嵌入的分类方法
3. 实现零样本分类
4. 多标签分类实现
5. 置信度评估和阈值设置
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

class TextClassificationSystem:
    """文本分类系统"""
    
    def __init__(self):
        """初始化分类系统"""
        self.client = EmbeddingClient()
        self.category_embeddings = {}
        self.training_data = {}
        
    def load_sample_data(self) -> Dict[str, List[str]]:
        """加载示例分类数据"""
        sample_data = {
            "科技": [
                "人工智能技术取得重大突破，深度学习算法性能提升显著",
                "苹果公司发布新一代芯片，计算能力提升50%",
                "量子计算机研究获得新进展，有望解决复杂问题",
                "5G网络技术推动物联网应用快速发展"
            ],
            "体育": [
                "国足在世界杯预选赛中表现出色，晋级下一轮",
                "NBA总决赛即将打响，湖人队和凯尔特人队争夺冠军",
                "中国女排在世界锦标赛中获得金牌，展现强大实力",
                "足球世界杯即将开幕，各支球队积极备战"
            ],
            "财经": [
                "央行宣布降息政策，刺激经济增长",
                "股票市场今日大涨，科技股领涨大盘",
                "房地产市场调控政策效果显著，房价趋于稳定",
                "国际贸易合作加强，双边贸易额创新高"
            ],
            "娱乐": [
                "电影《流浪地球3》票房突破10亿，创影史纪录",
                "某知名歌手发布新专辑，音乐风格大受好评",
                "电视剧《三体》获得观众一致好评，科幻题材受欢迎",
                "综艺节目创新形式，吸引大量年轻观众"
            ]
        }
        return sample_data
    
    def prepare_category_embeddings(self, categories: Dict[str, List[str]]):
        """准备类别嵌入向量"""
        print("🎯 准备类别嵌入...")
        
        for category, examples in categories.items():
            # 获取类别名称的嵌入
            category_embedding = self.client.get_embedding(category)
            
            # 获取示例文本的平均嵌入
            example_embeddings = []
            for example in examples:
                embedding = self.client.get_embedding(example)
                example_embeddings.append(embedding)
            
            # 计算类别中心向量
            category_center = np.mean(example_embeddings, axis=0)
            
            self.category_embeddings[category] = {
                'name': category,
                'embedding': category_center,
                'examples': examples
            }
            print(f"   ✅ {category} 类别已准备完成")
    
    def classify_text(self, text: str, threshold: float = 0.6) -> List[Dict[str, float]]:
        """对文本进行分类"""
        if not self.category_embeddings:
            raise ValueError("请先调用prepare_category_embeddings()准备类别数据")
        
        # 获取文本嵌入
        text_embedding = np.array(self.client.get_embedding(text))
        
        # 计算与每个类别的相似度
        similarities = []
        for category_name, category_data in self.category_embeddings.items():
            category_embedding = np.array(category_data['embedding'])
            similarity = np.dot(text_embedding, category_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(category_embedding)
            )
            similarities.append({
                'category': category_name,
                'similarity': similarity,
                'confidence': similarity
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 过滤低于阈值的分类
        filtered_results = [result for result in similarities if result['similarity'] >= threshold]
        
        return filtered_results
    
    def zero_shot_classification(self, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        """零样本分类"""
        print(f"\n🎯 零样本分类: '{text[:30]}...'")
        
        # 获取文本嵌入
        text_embedding = np.array(self.client.get_embedding(text))
        
        # 计算与每个候选标签的相似度
        label_similarities = {}
        for label in candidate_labels:
            label_embedding = np.array(self.client.get_embedding(label))
            similarity = np.dot(text_embedding, label_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(label_embedding)
            )
            label_similarities[label] = similarity
        
        # 按相似度排序
        sorted_labels = sorted(label_similarities.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_labels)
    
    def multi_label_classification(self, text: str, labels: List[str], threshold: float = 0.5) -> List[str]:
        """多标签分类"""
        print(f"\n🏷️ 多标签分类: '{text[:30]}...'")
        
        # 获取所有标签的相似度
        similarities = self.zero_shot_classification(text, labels)
        
        # 返回超过阈值的标签
        applicable_labels = [label for label, score in similarities.items() if score >= threshold]
        
        return applicable_labels
    
    def evaluate_classification_accuracy(self, test_data: Dict[str, List[str]]) -> Dict[str, float]:
        """评估分类准确率"""
        print("\n📊 评估分类准确率...")
        
        correct_predictions = 0
        total_predictions = 0
        category_stats = {}
        
        for true_category, test_texts in test_data.items():
            category_stats[true_category] = {'correct': 0, 'total': 0}
            
            for test_text in test_texts:
                # 分类文本
                predictions = self.classify_text(test_text, threshold=0.3)
                
                if predictions:
                    predicted_category = predictions[0]['category']
                    
                    if predicted_category == true_category:
                        correct_predictions += 1
                        category_stats[true_category]['correct'] += 1
                    
                    total_predictions += 1
                    category_stats[true_category]['total'] += 1
        
        # 计算总体准确率
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 计算各类别准确率
        category_accuracies = {}
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                category_accuracies[category] = stats['correct'] / stats['total']
        
        return {
            'overall_accuracy': overall_accuracy,
            'category_accuracies': category_accuracies,
            'category_stats': category_stats
        }
    
    def demonstrate_confidence_thresholding(self):
        """演示置信度阈值设置"""
        print("\n🎯 第3部分：置信度阈值设置")
        print("=" * 50)
        
        test_texts = [
            "苹果公司发布新一代iPhone，搭载最新芯片技术",
            "国足在亚洲杯比赛中表现出色，球迷热情高涨",
            "央行宣布新的货币政策，影响股市和房地产市场",
            "某知名演员发布新歌，音乐风格创新独特"
        ]
        
        thresholds = [0.3, 0.5, 0.7, 0.9]
        
        for text in test_texts:
            print(f"\n📄 测试文本: {text}")
            for threshold in thresholds:
                results = self.classify_text(text, threshold=threshold)
                if results:
                    category = results[0]['category']
                    confidence = results[0]['confidence']
                    print(f"   阈值 {threshold}: {category} (置信度: {confidence:.3f})")
                else:
                    print(f"   阈值 {threshold}: 无匹配类别")
    
    def demo_text_classification(self):
        """演示文本分类功能"""
        print("🚀 文本分类系统演示")
        print("=" * 60)
        
        # 加载示例数据
        sample_data = self.load_sample_data()
        
        # 准备类别嵌入
        self.prepare_category_embeddings(sample_data)
        
        # 演示分类功能
        test_texts = [
            "人工智能技术正在改变医疗行业的诊断方式",
            "足球队在世界杯预选赛中获得重要胜利",
            "央行宣布新的金融政策，刺激经济增长",
            "新电影票房突破记录，观众反响热烈"
        ]
        
        print("\n🎯 第1部分：文本分类演示")
        print("=" * 50)
        
        for text in test_texts:
            print(f"\n📄 文本: {text}")
            results = self.classify_text(text, threshold=0.5)
            
            if results:
                print(f"   🏷️ 分类结果:")
                for result in results[:3]:  # 显示前3个结果
                    print(f"      {result['category']}: {result['confidence']:.3f}")
            else:
                print("   ❌ 无法分类")
        
        # 演示零样本分类
        print("\n🎯 第2部分：零样本分类演示")
        print("=" * 50)
        
        zero_shot_text = "某科技公司宣布开发新型量子计算芯片"
        candidate_labels = ["科技", "体育", "财经", "娱乐", "教育", "医疗"]
        
        zero_shot_results = self.zero_shot_classification(zero_shot_text, candidate_labels)
        print(f"\n📄 文本: {zero_shot_text}")
        print("🔍 零样本分类结果:")
        for label, score in list(zero_shot_results.items())[:3]:
            print(f"   {label}: {score:.3f}")
        
        # 演示多标签分类
        print("\n🎯 第3部分：多标签分类演示")
        print("=" * 50)
        
        multi_label_text = "人工智能技术应用于医疗诊断，提高疾病检测准确率"
        multi_labels = ["科技", "医疗", "教育", "商业", "研究"]
        
        multi_results = self.multi_label_classification(multi_label_text, multi_labels)
        print(f"\n📄 文本: {multi_label_text}")
        print("🏷️ 多标签分类结果:")
        for label in multi_results:
            print(f"   ✅ {label}")
        
        # 演示置信度阈值
        self.demonstrate_confidence_thresholding()
        
        # 评估分类准确率
        print("\n🎯 第4部分：分类准确率评估")
        print("=" * 50)
        
        # 创建测试数据
        test_data = {
            "科技": [
                "新型人工智能技术突破传统算法限制",
                "5G网络技术推动物联网快速发展",
                "量子计算机研究取得重要进展"
            ],
            "体育": [
                "世界杯足球赛即将开幕，各队积极备战",
                "NBA季后赛竞争激烈，多支球队有望夺冠",
                "奥运会筹备工作进展顺利，场馆建设完成"
            ],
            "财经": [
                "股市今日大涨，科技股领涨市场",
                "央行宣布降息政策，刺激经济增长",
                "房地产市场调控政策效果显著"
            ]
        }
        
        evaluation_results = self.evaluate_classification_accuracy(test_data)
        print(f"总体准确率: {evaluation_results['overall_accuracy']:.2%}")
        
        for category, accuracy in evaluation_results['category_accuracies'].items():
            print(f"{category}: {accuracy:.2%}")

def main():
    """主函数"""
    print("🚀 文本分类系统")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        classifier = TextClassificationSystem()
        classifier.demo_text_classification()
        
        print("\n🎉 文本分类演示完成！")
        print("\n核心技术总结:")
        print("   • 基于嵌入的文本分类")
        print("   • 零样本分类")
        print("   • 多标签分类")
        print("   • 置信度评估")
        print("\n实际应用场景:")
        print("   • 新闻文章自动分类")
        print("   • 垃圾邮件检测")
        print("   • 客户反馈分析")
        print("   • 社交媒体内容审核")
        print("\n下一课：03-recommendation-system.py - 推荐系统")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
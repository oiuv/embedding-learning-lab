#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1课：什么是文本嵌入 (Text Embedding)
========================================

本课程将帮助你理解文本嵌入的基本概念和工作原理。

学习目标：
1. 理解什么是文本嵌入
2. 了解文本嵌入的用途
3. 认识向量空间模型
4. 理解语义相似性

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

class TextEmbeddingConcept:
    """文本嵌入概念演示类"""
    
    def __init__(self):
        """初始化概念演示"""
        self.concepts = {
            "word_to_vector": "将单词映射到数值向量的过程",
            "semantic_space": "高维空间中的语义表示",
            "similarity": "通过向量距离计算语义相似性",
            "dimension": "向量的长度，通常128-1024维"
        }
    
    def demonstrate_word_mapping(self):
        """演示单词到向量的映射"""
        print("🎯 第1部分：单词到向量的映射")
        print("=" * 50)
        
        # 简单的单词到向量映射示例（模拟）
        word_vectors = {
            "猫": [0.1, 0.3, 0.8, 0.2],
            "狗": [0.2, 0.4, 0.7, 0.3],
            "汽车": [0.8, 0.1, 0.0, 0.9],
            "飞机": [0.9, 0.2, 0.1, 0.8]
        }
        
        print("📊 单词 -> 向量映射示例:")
        for word, vector in word_vectors.items():
            print(f"'{word}' -> {vector}")
        
        return word_vectors
    
    def demonstrate_similarity_calculation(self, word_vectors: Dict[str, List[float]]):
        """演示相似度计算"""
        print("\n🎯 第2部分：计算单词相似度")
        print("=" * 50)
        
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """计算余弦相似度"""
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        
        # 计算相似度
        words = list(word_vectors.keys())
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
                print(f"'{word1}' 和 '{word2}' 的相似度: {sim:.3f}")
    
    def visualize_2d_projection(self):
        """可视化2D投影"""
        print("\n🎯 第3部分：2D可视化")
        print("=" * 50)
        
        # 创建示例数据
        words = ["高兴", "悲伤", "愤怒", "惊讶", "恐惧", "厌恶"]
        # 模拟情感向量的2D投影
        coordinates = {
            "高兴": [0.8, 0.7],
            "悲伤": [-0.7, -0.5],
            "愤怒": [-0.5, 0.8],
            "惊讶": [0.6, -0.3],
            "恐惧": [-0.8, -0.2],
            "厌恶": [-0.4, -0.8]
        }
        
        # 绘制2D图
        # 设置matplotlib以支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 8))
        
        for word, (x, y) in coordinates.items():
            plt.scatter(x, y, s=100)
            plt.annotate(word, (x, y), fontsize=12, ha='center', va='bottom')
        
        plt.title("情感词汇的2D向量空间表示", fontsize=14)
        plt.xlabel("维度1", fontsize=12)
        plt.ylabel("维度2", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 保存图片
        plt.savefig('01-basics/embedding_concept_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可视化已保存为 'embedding_concept_visualization.png'")
    
    def explain_key_concepts(self):
        """解释关键概念"""
        print("\n🎯 第4部分：关键概念解释")
        print("=" * 50)
        
        concepts = {
            "维度 (Dimension)": 
                "文本向量的长度，128维表示每个文本用128个数字表示",
            "语义空间 (Semantic Space)": 
                "高维空间，语义相似的文本在空间中距离较近",
            "余弦相似度 (Cosine Similarity)": 
                "通过向量夹角计算相似度，范围-1到1，1表示完全相同",
            "嵌入模型 (Embedding Model)": 
                "将文本转换为向量的AI模型，如Word2Vec、BERT等",
            "上下文 (Context)": 
                "文本的环境信息，影响词的含义"
        }
        
        for concept, explanation in concepts.items():
            print(f"📖 {concept}:")
            print(f"   {explanation}\n")
    
    def demonstrate_real_world_examples(self):
        """演示实际应用场景"""
        print("🎯 第5部分：实际应用场景")
        print("=" * 50)
        
        scenarios = [
            {
                "场景": "搜索引擎",
                "描述": "理解用户搜索意图，返回相关结果",
                "示例": "搜索'苹果'可区分水果和公司"
            },
            {
                "场景": "推荐系统",
                "描述": "基于内容相似性推荐商品或文章",
                "示例": "阅读机器学习文章后推荐AI相关内容"
            },
            {
                "场景": "情感分析",
                "描述": "分析文本的情感倾向",
                "示例": "判断用户评论是正面还是负面"
            },
            {
                "场景": "智能客服",
                "描述": "理解用户问题，匹配最佳答案",
                "示例": "'如何退货'匹配到退货政策"
            },
            {
                "场景": "内容聚类",
                "描述": "将相似内容自动分组",
                "示例": "将新闻文章按主题分类"
            }
        ]
        
        for scenario in scenarios:
            print(f"🎯 {scenario['场景']}:")
            print(f"   描述: {scenario['描述']}")
            print(f"   示例: {scenario['示例']}\n")

def main():
    """主函数"""
    print("🚀 第1课：什么是文本嵌入")
    print("=" * 60)
    print("欢迎来到文本嵌入学习之旅！")
    print("本课程将帮助你理解文本嵌入的核心概念。\n")
    
    try:
        input("📚 按回车键开始学习...")
        
        # 创建演示实例
        demo = TextEmbeddingConcept()
        
        # 运行演示
        print("\n" + "="*60)
        word_vectors = demo.demonstrate_word_mapping()
        input("\n📚 按回车键继续到相似度计算...")
        
        print("\n" + "="*60)
        demo.demonstrate_similarity_calculation(word_vectors)
        input("\n📊 按回车键继续到可视化部分...")
        
        print("\n" + "="*60)
        demo.visualize_2d_projection()
        input("\n📖 按回车键继续到关键概念解释...")
        
        print("\n" + "="*60)
        demo.explain_key_concepts()
        input("\n🎯 按回车键查看实际应用场景...")
        
        print("\n" + "="*60)
        demo.demonstrate_real_world_examples()
        
        print("\n" + "="*60)
        print("🎉 第1课完成！")
        print("你已经了解了：")
        print("✅ 什么是文本嵌入")
        print("✅ 如何计算文本相似度")
        print("✅ 文本嵌入的实际应用")
        print("\n📂 可视化图片已保存为 'embedding_concept_visualization.png'")
        print("\n🎯 下一课：02-first-embedding.py - 获取第一个文本向量")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {str(e)}")
        print("🔄 请检查环境配置后重试")
    finally:
        input("\n📚 按回车键退出课程...")

if __name__ == "__main__":
    main()
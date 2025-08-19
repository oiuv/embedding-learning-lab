#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级教程第5课：文本排序模型基础
============================

本课程将教你如何使用文本排序模型(gte-rerank)优化搜索结果和问答系统。

学习目标：
1. 理解文本排序模型的原理和应用场景
2. 掌握gte-rerank模型的使用方法
3. 学会集成排序模型到现有系统
4. 理解排序模型与嵌入模型的区别
5. 实现复杂场景的排序优化

"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

# 初始化DashScope
import dashscope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

@dataclass
class RankedDocument:
    """排序后的文档"""
    text: str
    score: float
    original_rank: int
    rerank_score: float
    metadata: Dict = None

class TextRerankTutorial:
    """文本排序教程类"""
    
    def __init__(self):
        """初始化教程"""
        self.client = EmbeddingClient()
        print("🚀 文本排序模型教程启动！")
        print("=" * 60)
    
    def basic_rerank_demo(self):
        """基础排序演示"""
        print("📚 基础文本排序演示")
        print("=" * 40)
        
        # 示例查询和文档
        query = "人工智能在医疗领域的最新应用"
        documents = [
            "人工智能在医疗影像诊断中的应用越来越广泛，特别是在CT和MRI分析方面",
            "机器学习算法可以帮助医生更准确地诊断疾病，提高诊断效率和准确性",
            "深度学习技术在医疗领域的应用包括疾病预测、药物研发和个性化治疗",
            "区块链技术在金融领域的应用主要集中在数字货币和智能合约方面",
            "人工智能在自动驾驶汽车中的应用涉及计算机视觉、路径规划和决策系统",
            "医疗机器人在手术中的应用提高了手术精度，减少了人为错误",
            "自然语言处理技术可以用于分析医疗记录和患者反馈信息",
            "量子计算在药物分子设计中的应用可能会加速新药研发进程",
            "人工智能辅助诊断系统可以帮助放射科医生检测早期癌症病变",
            "远程医疗技术结合AI可以为偏远地区提供更好的医疗服务"
        ]
        
        print(f"🔍 查询: {query}")
        print(f"📄 文档数量: {len(documents)}")
        
        # 1. 先使用嵌入模型获取初始排序
        print("\n1️⃣ 使用嵌入模型获取初始排序...")
        query_embedding = self.client.get_embedding(query)
        doc_embeddings = [self.client.get_embedding(doc) for doc in documents]
        
        # 计算余弦相似度
        similarities = []
        for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings)):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((i, doc, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        print("\n📊 嵌入模型排序结果:")
        for rank, (idx, doc, score) in enumerate(similarities[:5], 1):
            print(f"   {rank}. 分数: {score:.3f} - {doc[:60]}...")
        
        # 2. 使用文本排序模型重新排序
        print("\n2️⃣ 使用文本排序模型重新排序...")
        try:
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=documents,
                top_n=5,
                return_documents=True
            )
            
            if response.status_code == 200:
                print("\n🎯 文本排序模型结果:")
                for rank, result in enumerate(response.output.results, 1):
                    print(f"   {rank}. 分数: {result.relevance_score:.3f} - {result.document[:60]}...")
                    
                    # 对比原始排名和新排名
                    original_idx = result.index
                    original_rank = next(i for i, (idx, _, _) in enumerate(similarities) if idx == original_idx) + 1
                    print(f"       📈 从第{original_rank}名提升到第{rank}名")
            else:
                print(f"❌ 排序失败: {response}")
                
        except Exception as e:
            print(f"❌ 调用排序模型失败: {e}")
    
    def complex_scenario_demo(self):
        """复杂场景演示"""
        print("\n🏥 医疗领域复杂场景演示")
        print("=" * 50)
        
        # 模拟医疗文献搜索场景
        query = "肺癌早期诊断的最新技术进展"
        
        # 包含各种相关度的医疗文献
        medical_docs = [
            {
                "title": "低剂量CT筛查在早期肺癌诊断中的应用价值",
                "content": "研究表明，低剂量螺旋CT筛查可以检测到直径小于1cm的肺结节，显著提高早期肺癌的检出率。",
                "type": "临床研究",
                "year": 2023,
                "citations": 156
            },
            {
                "title": "人工智能辅助诊断系统在肺癌筛查中的meta分析",
                "content": "通过对12项随机对照试验的meta分析，发现AI辅助诊断系统可以提高肺癌筛查的敏感性和特异性。",
                "type": "系统综述",
                "year": 2024,
                "citations": 89
            },
            {
                "title": "液体活检技术在肺癌早期检测中的突破",
                "content": "循环肿瘤DNA(ctDNA)检测技术为肺癌早期诊断提供了新的无创检测方法，特别适用于高风险人群筛查。",
                "type": "基础研究",
                "year": 2023,
                "citations": 234
            },
            {
                "title": "PET-CT在肺癌分期中的诊断准确性研究",
                "content": "虽然PET-CT在肺癌分期中具有重要作用，但其在早期病变检测中的敏感性仍有待提高。",
                "type": "影像学研究",
                "year": 2022,
                "citations": 78
            },
            {
                "title": "肺癌流行病学调查及危险因素分析",
                "content": "吸烟仍是肺癌最主要的危险因素，但环境污染和遗传因素的作用日益受到关注。",
                "type": "流行病学研究",
                "year": 2023,
                "citations": 312
            },
            {
                "title": "免疫治疗在晚期肺癌中的疗效评估",
                "content": "PD-1/PD-L1抑制剂显著改善了晚期非小细胞肺癌患者的生存期，但生物标志物选择仍是挑战。",
                "type": "临床试验",
                "year": 2024,
                "citations": 445
            },
            {
                "title": "机器学习在病理切片肺癌诊断中的应用",
                "content": "深度学习算法在肺癌病理切片分析中表现出与病理专家相当的诊断准确性，有望提高诊断效率。",
                "type": "人工智能应用",
                "year": 2023,
                "citations": 167
            },
            {
                "title": "肺癌筛查的成本效益分析",
                "content": "从公共卫生角度分析，针对高风险人群的肺癌筛查项目具有良好的成本效益比。",
                "type": "卫生经济学研究",
                "year": 2023,
                "citations": 93
            }
        ]
        
        # 构建文档文本
        documents = [f"{doc['title']}. {doc['content']}" for doc in medical_docs]
        
        print(f"🔍 医疗查询: {query}")
        print(f"📊 文档类型: {len(set([doc['type'] for doc in medical_docs]))}种")
        print(f"📅 时间跨度: 2022-2024年")
        
        # 执行排序
        try:
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=documents,
                top_n=5,
                return_documents=True
            )
            
            if response.status_code == 200:
                print("\n🏆 医疗文献排序结果:")
                for rank, result in enumerate(response.output.results, 1):
                    doc_idx = result.index
                    doc_info = medical_docs[doc_idx]
                    
                    print(f"\n   {rank}. 📄 {doc_info['title']}")
                    print(f"       📊 相关性分数: {result.relevance_score:.3f}")
                    print(f"       🏷️ 类型: {doc_info['type']}")
                    print(f"       📅 年份: {doc_info['year']}")
                    print(f"       📈 引用数: {doc_info['citations']}")
                    print(f"       📝 {doc_info['content'][:100]}...")
                    
        except Exception as e:
            print(f"❌ 医疗场景演示失败: {e}")
    
    def multilingual_demo(self):
        """多语言排序演示"""
        print("\n🌏 多语言文本排序演示")
        print("=" * 50)
        
        # 同一查询的多种语言版本
        queries = {
            "zh": "人工智能在医疗诊断中的应用",
            "en": "Applications of AI in medical diagnosis",
            "ja": "医療診断におけるAIの応用",
            "ko": "의료 진단에서 AI의 응용"
        }
        
        # 多语言文档
        multilingual_docs = [
            "人工智能在医疗影像分析中的应用越来越广泛，特别是在CT和MRI诊断方面",
            "AI applications in medical imaging analysis are becoming increasingly widespread, especially in CT and MRI diagnostics",
            "医療画像解析におけるAIの応用は、CTやMRI診断においてますます広がっています",
            "의료 영상 분석에서 AI의 응용이 점점 더 널리 퍼지고 있으며, 특히 CT 및 MRI 진단에서",
            "机器学习算法可以帮助医生更准确地诊断疾病，提高诊断效率和准确性",
            "Machine learning algorithms can help doctors diagnose diseases more accurately, improving diagnostic efficiency and accuracy",
            "機械学習アルゴリズムは医師が病気をより正確に診断するのを助け、診断効率と精度を向上させることができます",
            "머신러닝 알고리즘은 의사들이 질병을 더 정확하게 진단하도록 도와 진단 효율성과 정확성을 향상시킵니다"
        ]
        
        for lang, query in queries.items():
            print(f"\n🔍 {lang.upper()} 查询: {query}")
            
            try:
                response = dashscope.TextReRank.call(
                    model="gte-rerank-v2",
                    query=query,
                    documents=multilingual_docs,
                    top_n=3,
                    return_documents=True
                )
                
                if response.status_code == 200:
                    print(f"   🎯 前3个最相关文档:")
                    for rank, result in enumerate(response.output.results, 1):
                        print(f"      {rank}. {result.relevance_score:.3f} - {result.document[:80]}...")
                        
            except Exception as e:
                print(f"   ❌ {lang} 语言演示失败: {e}")
    
    def performance_comparison(self):
        """性能对比分析"""
        print("\n⚡ 性能对比分析")
        print("=" * 40)
        
        # 准备测试数据
        test_query = "人工智能发展趋势"
        test_docs = [
            f"这是关于AI的第{i}篇文档，讨论了人工智能在不同领域的应用和发展前景。"
            f"特别关注了机器学习、深度学习和自然语言处理技术的最新进展。"
            for i in range(1, 21)
        ]
        
        # 测试不同规模下的性能
        sizes = [5, 10, 15, 20]
        
        print("📊 不同数据规模下的性能对比:")
        print("规模 | 嵌入时间 | 排序时间 | 总时间")
        print("-" * 40)
        
        for size in sizes:
            docs_subset = test_docs[:size]
            
            # 嵌入模型时间
            start_time = time.time()
            query_emb = self.client.get_embedding(test_query)
            doc_embs = [self.client.get_embedding(doc) for doc in docs_subset]
            
            similarities = []
            for emb in doc_embs:
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                similarities.append(sim)
            
            embedding_time = time.time() - start_time
            
            # 文本排序时间
            start_time = time.time()
            try:
                response = dashscope.TextReRank.call(
                    model="gte-rerank-v2",
                    query=test_query,
                    documents=docs_subset,
                    top_n=size
                )
                rerank_time = time.time() - start_time
                
                total_time = embedding_time + rerank_time
                
                print(f"{size:4d} | {embedding_time:8.3f}s | {rerank_time:8.3f}s | {total_time:8.3f}s")
                
            except Exception as e:
                print(f"{size:4d} | {embedding_time:8.3f}s | {'ERROR':8s} | {'N/A':8s}")
    
    def run_tutorial(self):
        """运行完整教程"""
        print("🎓 文本排序模型完整教程")
        print("=" * 60)
        
        # 检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
            return
        
        try:
            # 运行各个演示
            self.basic_rerank_demo()
            self.complex_scenario_demo()
            self.multilingual_demo()
            self.performance_comparison()
            
            print("\n🎉 文本排序模型教程完成！")
            print("\n📚 你学会了：")
            print("✅ 文本排序模型的基本原理")
            print("✅ gte-rerank模型的使用方法")
            print("✅ 复杂场景下的排序优化")
            print("✅ 多语言文本排序")
            print("✅ 性能评估和对比分析")
            
            print("\n📖 下一步学习：")
            print("   1. 阅读02-integration-guide.py - 系统集成指南")
            print("   2. 查看03-advanced-techniques.py - 高级技巧")
            print("   4. 运行04-real-world-examples.py - 实际案例")
            
        except Exception as e:
            print(f"❌ 教程运行失败: {e}")

def main():
    """主函数"""
    tutorial = TextRerankTutorial()
    tutorial.run_tutorial()

if __name__ == "__main__":
    main()
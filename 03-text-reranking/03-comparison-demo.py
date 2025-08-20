#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级教程第7课：排序模型对比演示
==============================

本课程将对比展示文本排序模型与现有嵌入模型的差异和优势。

学习目标：
1. 理解不同排序方法的优缺点
2. 掌握性能对比分析方法
3. 学会选择合适的排序策略
4. 分析排序结果的质量差异
5. 优化排序参数配置

"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 设置中文字体 - 优先支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 数学符号警告可以忽略，不影响中文显示

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient
from utils.text_reranker import TextReranker, RerankDocument, RerankResult

# 初始化DashScope
import dashscope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

@dataclass
class ComparisonMetrics:
    """对比指标"""
    method: str
    query: str
    top_k: int
    precision: float
    recall: float
    f1_score: float
    avg_precision: float
    ndcg: float
    processing_time: float
    cost_estimate: float

class SortingComparison:
    """排序对比分析器"""
    
    def __init__(self):
        """初始化对比分析器"""
        self.embedding_client = EmbeddingClient()
        self.reranker = TextReranker()
        
        # 评估数据
        self.evaluation_queries = [
            {
                "query": "人工智能在医疗诊断中的应用",
                "relevant_docs": [0, 1, 2, 5, 8],
                "description": "医疗AI应用"
            },
            {
                "query": "区块链技术在金融支付中的应用",
                "relevant_docs": [3, 6, 9, 12, 15],
                "description": "区块链金融应用"
            },
            {
                "query": "深度学习算法的最新进展",
                "relevant_docs": [1, 4, 7, 10, 13],
                "description": "深度学习进展"
            }
        ]
        
        # 生成测试文档
        self.test_documents = self._generate_test_corpus()
        
        print("🔍 排序模型对比分析启动")
        print("=" * 50)
    
    def _generate_test_corpus(self) -> List[RerankDocument]:
        """生成测试文档语料"""
        documents = [
            # 医疗AI相关
            "人工智能在医疗影像诊断中的应用越来越广泛，特别是在CT和MRI分析方面，准确率已达到95%以上",
            "机器学习算法可以帮助医生更准确地诊断疾病，通过分析患者的历史数据提高诊断效率和准确性",
            "深度学习技术在医疗领域的应用包括疾病预测、药物研发和个性化治疗方案制定",
            # 区块链相关
            "区块链技术为金融行业带来了革命性变化，特别是在跨境支付和数字货币交易方面",
            "智能合约技术使得金融交易更加透明和高效，减少了中间环节和交易成本",
            "去中心化金融(DeFi)应用基于区块链技术，为用户提供了无需传统银行的金融服务",
            # 深度学习相关
            "深度学习算法在图像识别领域取得了突破性进展，准确率已经超过人类水平",
            "神经网络架构的优化使得深度学习模型在计算效率和性能方面都有显著提升",
            "Transformer架构的出现彻底改变了自然语言处理领域的技术发展方向",
            # 其他相关
            "云计算技术为企业提供了灵活可扩展的计算资源，降低了IT基础设施成本",
            "大数据分析帮助企业从海量数据中提取有价值的商业洞察和市场趋势",
            "物联网技术正在连接数十亿设备，构建智能化的生活和工作环境",
            "计算机视觉技术在自动驾驶汽车中的应用日益成熟，安全性不断提升",
            "自然语言处理技术使得计算机能够理解和生成人类语言，支持多语言交流",
            "机器学习在金融风险评估中的应用显著提高了风险预测的准确性和及时性",
            "人工智能辅助诊断系统可以帮助放射科医生检测早期癌症病变，提高诊断精度",
            "远程医疗技术结合AI可以为偏远地区提供更好的医疗服务，缩小医疗差距"
        ]
        
        return [
            RerankDocument(
                text=text,
                doc_id=f"doc_{i}",
                score=np.random.uniform(0.3, 0.8),
                metadata={
                    "index": i,
                    "length": len(text),
                    "category": "tech" if i < 10 else "medical" if i < 13 else "finance"
                }
            )
            for i, text in enumerate(documents)
        ]
    
    def _embedding_based_ranking(self, 
                                query: str, 
                                documents: List[RerankDocument],
                                top_k: int = 10) -> List[Tuple[int, float]]:
        """基于嵌入的排序"""
        # 获取查询嵌入
        query_embedding = self.embedding_client.get_embedding(query)
        
        # 获取文档嵌入和计算相似度
        similarities = []
        for i, doc in enumerate(documents):
            doc_embedding = self.embedding_client.get_embedding(doc.text)
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _keyword_based_ranking(self, 
                             query: str, 
                             documents: List[RerankDocument],
                             top_k: int = 10) -> List[Tuple[int, float]]:
        """基于关键词的排序"""
        query_words = set(query.lower().split())
        
        scores = []
        for i, doc in enumerate(documents):
            doc_words = set(doc.text.lower().split())
            
            # 计算关键词匹配度
            intersection = query_words.intersection(doc_words)
            score = len(intersection) / len(query_words) if query_words else 0
            
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _rerank_based_ranking(self, 
                            query: str, 
                            documents: List[RerankDocument],
                            top_k: int = 10) -> List[Tuple[int, float]]:
        """基于文本排序模型的排序"""
        try:
            results = self.reranker.rerank(query, documents, top_n=top_k)
            if results:
                return [(r.original_rank, r.relevance_score) for r in results]
            else:
                print("⚠️ 文本排序模型未返回有效结果，使用嵌入模型作为备选")
                return self._embedding_based_ranking(query, documents, top_k)
        except Exception as e:
            print(f"⚠️ 文本排序模型调用失败: {e}，使用嵌入模型作为备选")
            return self._embedding_based_ranking(query, documents, top_k)
    
    def _calculate_metrics(self, 
                          ranked_indices: List[Tuple[int, float]],
                          relevant_indices: List[int],
                          k: int = 10) -> Dict[str, float]:
        """计算排序指标"""
        # 获取前k个结果的索引
        top_k_indices = [idx for idx, _ in ranked_indices[:k]]
        
        # 计算精确率
        relevant_retrieved = len(set(top_k_indices).intersection(set(relevant_indices)))
        precision = relevant_retrieved / k if k > 0 else 0
        
        # 计算召回率
        recall = relevant_retrieved / len(relevant_indices) if relevant_indices else 0
        
        # 计算F1分数
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算平均精确率(AP)
        ap_sum = 0
        relevant_count = 0
        for i, idx in enumerate(top_k_indices):
            if idx in relevant_indices:
                relevant_count += 1
                ap_sum += relevant_count / (i + 1)
        
        avg_precision = ap_sum / len(relevant_indices) if relevant_indices else 0
        
        # 计算NDCG
        dcg = 0
        idcg = 0
        for i, idx in enumerate(top_k_indices):
            relevance = 1 if idx in relevant_indices else 0
            dcg += relevance / np.log2(i + 2)
        
        # 计算理想DCG
        ideal_relevances = [1] * min(len(relevant_indices), k) + [0] * max(0, k - len(relevant_indices))
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_precision": avg_precision,
            "ndcg": ndcg
        }
    
    def compare_sorting_methods(self) -> List[ComparisonMetrics]:
        """对比不同排序方法"""
        print("📊 开始排序方法对比分析...")
        
        results = []
        
        for test_case in self.evaluation_queries:
            query = test_case["query"]
            relevant_docs = test_case["relevant_docs"]
            
            print(f"\n🔍 分析查询: {query}")
            
            # 方法1: 嵌入模型排序
            start_time = time.time()
            embedding_ranked = self._embedding_based_ranking(query, self.test_documents)
            embedding_time = time.time() - start_time
            
            embedding_metrics = self._calculate_metrics(embedding_ranked, relevant_docs)
            
            # 方法2: 关键词排序
            start_time = time.time()
            keyword_ranked = self._keyword_based_ranking(query, self.test_documents)
            keyword_time = time.time() - start_time
            
            keyword_metrics = self._calculate_metrics(keyword_ranked, relevant_docs)
            
            # 方法3: 文本排序模型
            start_time = time.time()
            rerank_ranked = self._rerank_based_ranking(query, self.test_documents)
            rerank_time = time.time() - start_time
            
            rerank_metrics = self._calculate_metrics(rerank_ranked, relevant_docs)
            
            # 成本估算
            embedding_cost = self.reranker.estimate_cost(len(self.test_documents))["estimated_cost_cny"]
            rerank_cost = embedding_cost * 1.2  # 文本排序稍贵
            
            # 创建结果
            methods = ["embedding", "keyword", "rerank"]
            times = [embedding_time, keyword_time, rerank_time]
            costs = [embedding_cost, 0, rerank_cost]  # 关键词几乎无成本
            metrics_list = [embedding_metrics, keyword_metrics, rerank_metrics]
            
            for method, time_cost, cost, metrics in zip(methods, times, costs, metrics_list):
                result = ComparisonMetrics(
                    method=method,
                    query=query,
                    top_k=10,
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    avg_precision=metrics["avg_precision"],
                    ndcg=metrics["ndcg"],
                    processing_time=time_cost,
                    cost_estimate=cost
                )
                results.append(result)
        
        return results
    
    def visualize_results(self, results: List[ComparisonMetrics]):
        """可视化对比结果"""
        print("📈 生成可视化图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        methods = ["嵌入模型", "关键词", "文本排序"]
        original_methods = ["embedding", "keyword", "rerank"]
        metrics = ["precision", "recall", "f1_score", "ndcg"]
        chinese_metrics = ["精确率", "召回率", "F1分数", "NDCG"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("排序方法性能对比分析", fontsize=16)
        
        for i, (metric, chinese_metric) in enumerate(zip(metrics, chinese_metrics)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # 计算每个方法的平均指标
            method_scores = {}
            for method, original_method in zip(methods, original_methods):
                scores = [getattr(r, metric) for r in results if r.method == original_method]
                method_scores[method] = np.mean(scores)
            
            # 绘制柱状图
            bars = ax.bar(methods, [method_scores[m] for m in methods])
            ax.set_title(f"{chinese_metric} 对比")
            ax.set_ylabel(chinese_metric)
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, score in zip(bars, [method_scores[m] for m in methods]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{score:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"sorting_comparison_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 性能时间对比
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        method_times = {}
        for method, original_method in zip(methods, original_methods):
            times = [r.processing_time for r in results if r.method == original_method]
            method_times[method] = np.mean(times)
        
        bars = ax.bar(methods, [method_times[m] for m in methods], color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_title("平均处理时间对比")
        ax.set_ylabel("时间 (秒)")
        ax.set_yscale('log')
        
        for bar, time_val in zip(bars, [method_times[m] for m in methods]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                   f"{time_val:.3f}秒", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"sorting_performance_time_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_comparison(self, results: List[ComparisonMetrics]):
        """打印详细对比结果"""
        print("\n📊 详细对比结果")
        print("=" * 80)
        
        # 按查询分组
        queries = list(set(r.query for r in results))
        
        for query in queries:
            print(f"\n🔍 查询: {query}")
            print("-" * 60)
            
            query_results = [r for r in results if r.query == query]
            
            # 打印表格
            print(f"{'方法':<10} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'NDCG':>8} {'时间(s)':>8} {'成本(元)':>10}")
            print("-" * 60)
            
            for result in query_results:
                print(f"{result.method:<10} {result.precision:>8.3f} {result.recall:>8.3f} "
                      f"{result.f1_score:>8.3f} {result.ndcg:>8.3f} "
                      f"{result.processing_time:>8.3f} {result.cost_estimate:>10.4f}")
        
        # 计算平均值
        print(f"\n📈 平均性能指标")
        print("-" * 60)
        
        methods = ["embedding", "keyword", "rerank"]
        print(f"{'方法':<10} {'精确率':>8} {'召回率':>8} {'F1分数':>8} {'NDCG':>8} {'时间(s)':>8}")
        print("-" * 60)
        
        for method in methods:
            method_results = [r for r in results if r.method == method]
            avg_precision = np.mean([r.precision for r in method_results])
            avg_recall = np.mean([r.recall for r in method_results])
            avg_f1 = np.mean([r.f1_score for r in method_results])
            avg_ndcg = np.mean([r.ndcg for r in method_results])
            avg_time = np.mean([r.processing_time for r in method_results])
            
            print(f"{method:<10} {avg_precision:>8.3f} {avg_recall:>8.3f} "
                  f"{avg_f1:>8.3f} {avg_ndcg:>8.3f} {avg_time:>8.3f}")
    
    def performance_recommendations(self, results: List[ComparisonMetrics]) -> Dict[str, str]:
        """生成性能优化建议"""
        recommendations = {}
        
        # 分析各方法表现
        method_stats = {}
        for method in ["embedding", "keyword", "rerank"]:
            method_results = [r for r in results if r.method == method]
            
            method_stats[method] = {
                "avg_precision": np.mean([r.precision for r in method_results]),
                "avg_recall": np.mean([r.recall for r in method_results]),
                "avg_time": np.mean([r.processing_time for r in method_results]),
                "avg_cost": np.mean([r.cost_estimate for r in method_results])
            }
        
        # 生成建议
        if method_stats["rerank"]["avg_precision"] > method_stats["embedding"]["avg_precision"]:
            recommendations["accuracy"] = "文本排序模型在精确率方面表现更好，建议在精度要求高的场景使用"
        else:
            recommendations["accuracy"] = "嵌入模型已能提供良好的精确率，可考虑成本优化"
        
        if method_stats["rerank"]["avg_time"] > method_stats["embedding"]["avg_time"] * 2:
            recommendations["performance"] = "文本排序模型耗时较长，建议对实时性要求不高的场景使用"
        else:
            recommendations["performance"] = "文本排序模型性能可接受，可用于实时场景"
        
        if method_stats["keyword"]["avg_precision"] < 0.5:
            recommendations["keyword"] = "关键词排序效果较差，不建议作为主要排序方法"
        else:
            recommendations["keyword"] = "关键词排序可作为快速筛选的备选方案"
        
        return recommendations
    
    def run_comparison_analysis(self):
        """运行完整对比分析"""
        print("🔍 开始排序方法对比分析")
        print("=" * 60)
        
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
            return
        
        try:
            # 执行对比分析
            results = self.compare_sorting_methods()
            
            # 打印详细结果
            self.print_detailed_comparison(results)
            
            # 可视化结果
            self.visualize_results(results)
            
            # 生成建议
            recommendations = self.performance_recommendations(results)
            
            print(f"\n💡 优化建议:")
            for key, value in recommendations.items():
                print(f"   {key}: {value}")
            
            print(f"\n🎉 对比分析完成！")
            
            # 保存结果
            self._save_results(results)
            
        except Exception as e:
            print(f"❌ 对比分析失败: {e}")
    
    def _save_results(self, results: List[ComparisonMetrics]):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sorting_comparison_results_{timestamp}.json"
        
        # 转换为可序列化格式
        data = [
            {
                "method": r.method,
                "query": r.query,
                "top_k": r.top_k,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "avg_precision": r.avg_precision,
                "ndcg": r.ndcg,
                "processing_time": r.processing_time,
                "cost_estimate": r.cost_estimate
            }
            for r in results
        ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 结果已保存到: {filename}")

def main():
    """主函数"""
    analyzer = SortingComparison()
    analyzer.run_comparison_analysis()

if __name__ == "__main__":
    main()
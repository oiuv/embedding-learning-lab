#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级教程第8课：性能基准测试与优化
================================

本课程将深入分析文本排序模型的性能表现，并提供优化建议。

学习目标：
1. 建立完善的性能基准测试体系
2. 分析不同规模数据下的性能表现
3. 掌握缓存和批处理优化技巧
4. 监控API调用成本和效率
5. 设计自适应排序策略

"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.font_manager as fm

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient
from utils.text_reranker import TextReranker, RerankDocument, create_sample_documents

# 设置中文字体 - 优先支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 数学符号警告可以忽略，不影响中文显示

# 配置
import dashscope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    document_count: int
    query_count: int
    total_time: float
    avg_time_per_query: float
    avg_time_per_document: float
    throughput: float  # docs/sec
    cache_hit_rate: float
    api_calls: int
    estimated_cost: float
    memory_usage_mb: float
    error_rate: float

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        """初始化性能测试器"""
        self.embedding_client = EmbeddingClient()
        self.reranker = TextReranker()
        
        # 测试配置
        self.test_configs = [
            {"name": "小规模", "doc_count": 10, "query_count": 5},
            {"name": "中等规模", "doc_count": 50, "query_count": 10},
            {"name": "大规模", "doc_count": 100, "query_count": 20},
            {"name": "超大规模", "doc_count": 200, "query_count": 30}
        ]
        
        # 测试查询
        self.test_queries = [
            "人工智能在医疗诊断中的应用",
            "区块链技术在金融支付中的应用",
            "深度学习算法的最新进展",
            "云计算技术的成本优化",
            "物联网设备的安全防护"
        ]
        
        print("⚡ 性能基准测试启动")
        print("=" * 50)
    
    def generate_test_data(self, doc_count: int, query_count: int) -> Tuple[List[str], List[List[RerankDocument]]]:
        """生成测试数据"""
        queries = self.test_queries[:query_count]
        
        # 为每个查询生成不同的文档集
        documents_list = []
        for i in range(query_count):
            # 创建相关和不相关的混合文档
            relevant_docs = create_sample_documents(min(doc_count // 2, 10))
            irrelevant_docs = create_sample_documents(max(doc_count - len(relevant_docs), 5))
            
            # 修改不相关文档的内容
            for doc in irrelevant_docs:
                doc.text = f"这是一个完全不相关的文档: {doc.text[:50]}..."
            
            all_docs = relevant_docs + irrelevant_docs
            documents_list.append(all_docs)
        
        return queries, documents_list
    
    def benchmark_single_scale(self, config: Dict) -> BenchmarkResult:
        """测试单个规模"""
        print(f"🧪 测试规模: {config['name']} ({config['doc_count']}文档, {config['query_count']}查询)")
        
        # 生成测试数据
        queries, documents_list = self.generate_test_data(
            config["doc_count"], config["query_count"]
        )
        
        # 重置缓存
        self.reranker.clear_cache()
        
        # 开始测试
        start_time = time.time()
        api_calls = 0
        total_documents = 0
        
        for query, documents in zip(queries, documents_list):
            total_documents += len(documents)
            
            # 执行排序
            reranked = self.reranker.rerank(query, documents)
            api_calls += 1
        
        total_time = time.time() - start_time
        
        # 获取缓存统计
        cache_stats = self.reranker.get_cache_stats()
        
        # 计算成本
        cost = self.reranker.estimate_cost(total_documents)["estimated_cost_cny"]
        
        return BenchmarkResult(
            test_name=config["name"],
            document_count=config["doc_count"],
            query_count=config["query_count"],
            total_time=total_time,
            avg_time_per_query=total_time / config["query_count"],
            avg_time_per_document=total_time / total_documents,
            throughput=total_documents / total_time,
            cache_hit_rate=cache_stats["hit_rate"],
            api_calls=api_calls,
            estimated_cost=cost,
            memory_usage_mb=0,  # 简化处理
            error_rate=0  # 简化处理
        )
    
    def benchmark_all_scales(self) -> List[BenchmarkResult]:
        """测试所有规模"""
        results = []
        
        for config in self.test_configs:
            try:
                result = self.benchmark_single_scale(config)
                results.append(result)
                print(f"   ✅ 完成 - 总时间: {result.total_time:.2f}s, 吞吐量: {result.throughput:.1f} docs/sec")
            except Exception as e:
                print(f"   ❌ 测试失败: {e}")
                # 创建错误结果
                results.append(BenchmarkResult(
                    test_name=config["name"],
                    document_count=config["doc_count"],
                    query_count=config["query_count"],
                    total_time=0,
                    avg_time_per_query=0,
                    avg_time_per_document=0,
                    throughput=0,
                    cache_hit_rate=0,
                    api_calls=0,
                    estimated_cost=0,
                    memory_usage_mb=0,
                    error_rate=1.0
                ))
        
        return results
    
    def benchmark_cache_impact(self) -> Dict:
        """测试缓存影响"""
        print("\n💾 缓存性能测试")
        
        # 测试数据
        query = "人工智能在医疗诊断中的应用"
        documents = create_sample_documents(50)
        
        # 首次调用（无缓存）
        self.reranker.clear_cache()
        start_time = time.time()
        result1 = self.reranker.rerank(query, documents)
        first_time = time.time() - start_time
        
        # 第二次调用（有缓存）
        start_time = time.time()
        result2 = self.reranker.rerank(query, documents)
        cached_time = time.time() - start_time
        
        # 缓存统计
        cache_stats = self.reranker.get_cache_stats()
        
        return {
            "first_call_time": first_time,
            "cached_call_time": cached_time,
            "speedup_factor": first_time / cached_time if cached_time > 0 else 0,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["cache_size"]
        }
    
    def benchmark_concurrent_requests(self, 
                                    concurrent_requests: int = 5,
                                    doc_count: int = 20) -> Dict:
        """测试并发性能"""
        print(f"\n⚡ 并发测试 ({concurrent_requests}并发请求)")
        
        # 准备测试数据
        test_data = []
        for i in range(concurrent_requests):
            query = f"测试查询 {i+1}"
            documents = create_sample_documents(doc_count)
            test_data.append((query, documents))
        
        # 串行执行
        start_time = time.time()
        for query, documents in test_data:
            self.reranker.rerank(query, documents)
        serial_time = time.time() - start_time
        
        # 并发执行（模拟）
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self.reranker.rerank, query, documents)
                for query, documents in test_data
            ]
            results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        return {
            "serial_time": serial_time,
            "concurrent_time": concurrent_time,
            "concurrency_overhead": concurrent_time - serial_time,
            "throughput_improvement": serial_time / concurrent_time if concurrent_time > 0 else 0
        }
    
    def benchmark_batch_processing(self, batch_sizes: List[int] = [10, 25, 50, 100]) -> Dict:
        """测试批处理性能"""
        print("\n📦 批处理性能测试")
        
        base_query = "人工智能技术发展趋势"
        base_documents = create_sample_documents(100)
        
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(base_documents):
                continue
                
            documents = base_documents[:batch_size]
            
            start_time = time.time()
            result = self.reranker.rerank(base_query, documents)
            processing_time = time.time() - start_time
            
            cost = self.reranker.estimate_cost(batch_size)["estimated_cost_cny"]
            
            results[batch_size] = {
                "batch_size": batch_size,
                "processing_time": processing_time,
                "throughput": batch_size / processing_time,
                "cost_per_document": cost / batch_size,
                "cost_efficiency": batch_size / cost if cost > 0 else 0
            }
        
        return results
    
    def generate_optimization_report(self, results: List[BenchmarkResult]) -> Dict:
        """生成优化报告"""
        report = {
            "summary": {},
            "recommendations": [],
            "optimal_configs": {},
            "cost_analysis": {}
        }
        
        # 性能分析
        valid_results = [r for r in results if r.error_rate == 0]
        if not valid_results:
            report["summary"]["status"] = "所有测试失败"
            return report
        
        # 找出最佳配置
        best_throughput = max(r.throughput for r in valid_results)
        best_config = next(r for r in valid_results if r.throughput == best_throughput)
        
        report["summary"].update({
            "best_throughput": best_throughput,
            "best_config": best_config.test_name,
            "total_tested_configs": len(results),
            "successful_configs": len(valid_results)
        })
        
        # 成本分析
        total_cost = sum(r.estimated_cost for r in valid_results)
        avg_cost_per_query = np.mean([r.estimated_cost for r in valid_results])
        
        report["cost_analysis"].update({
            "total_test_cost": total_cost,
            "avg_cost_per_query": avg_cost_per_query,
            "cost_efficiency_threshold": 0.001  # 元/查询
        })
        
        # 生成优化建议
        if best_config.document_count <= 50:
            report["recommendations"].append(
                "小规模数据(≤50文档)可直接使用文本排序模型"
            )
        
        if best_config.cache_hit_rate > 0.5:
            report["recommendations"].append(
                "启用缓存可显著提升性能，建议缓存TTL设为1小时"
            )
        
        report["recommendations"].append(
            f"当前最佳配置: {best_config.test_name} - 吞吐量{best_throughput:.1f} docs/sec"
        )
        
        return report
    
    def visualize_benchmark_results(self, results: List[BenchmarkResult]):
        """可视化基准测试结果"""
        valid_results = [r for r in results if r.error_rate == 0]
        
        if not valid_results:
            print("❌ 无有效数据用于可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("文本排序性能基准测试结果", fontsize=16)
        
        # 数据准备
        scales = [r.test_name for r in valid_results]
        throughputs = [r.throughput for r in valid_results]
        times = [r.avg_time_per_query for r in valid_results]
        costs = [r.estimated_cost for r in valid_results]
        
        # 吞吐量对比
        axes[0, 0].bar(scales, throughputs, color='skyblue')
        axes[0, 0].set_title("吞吐量对比")
        axes[0, 0].set_ylabel("吞吐量 (文档/秒)")
        
        # 处理时间对比
        axes[0, 1].bar(scales, times, color='lightgreen')
        axes[0, 1].set_title("平均处理时间")
        axes[0, 1].set_ylabel("时间 (秒)")
        
        # 成本对比
        axes[1, 0].bar(scales, costs, color='salmon')
        axes[1, 0].set_title("预估成本 (元/查询)")
        axes[1, 0].set_ylabel("成本(元)")
        
        # 缓存命中率
        cache_rates = [r.cache_hit_rate for r in valid_results]
        axes[1, 1].bar(scales, cache_rates, color='gold')
        axes[1, 1].set_title("缓存命中率")
        axes[1, 1].set_ylabel("命中率")
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"benchmark_results_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_benchmark(self):
        """运行完整基准测试"""
        print("🚀 开始完整性能基准测试")
        print("=" * 60)
        
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
            return
        
        try:
            # 1. 基础规模测试
            print("\n1️⃣ 基础规模测试")
            scale_results = self.benchmark_all_scales()
            
            # 2. 缓存测试
            print("\n2️⃣ 缓存性能测试")
            cache_results = self.benchmark_cache_impact()
            
            # 3. 并发测试
            print("\n3️⃣ 并发性能测试")
            concurrent_results = self.benchmark_concurrent_requests()
            
            # 4. 批处理测试
            print("\n4️⃣ 批处理性能测试")
            batch_results = self.benchmark_batch_processing()
            
            # 5. 生成优化报告
            print("\n5️⃣ 生成优化报告")
            optimization_report = self.generate_optimization_report(scale_results)
            
            # 6. 可视化结果
            self.visualize_benchmark_results(scale_results)
            
            # 7. 保存完整结果
            self._save_benchmark_results({
                "scale_results": [asdict(r) for r in scale_results],
                "cache_results": cache_results,
                "concurrent_results": concurrent_results,
                "batch_results": batch_results,
                "optimization_report": optimization_report,
                "timestamp": datetime.now().isoformat()
            })
            
            print("\n📊 基准测试完成！")
            
        except Exception as e:
            print(f"❌ 基准测试失败: {e}")
    
    def _save_benchmark_results(self, results: Dict):
        """保存基准测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_benchmark_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 完整结果已保存到: {filename}")
    
    def quick_performance_check(self) -> Dict:
        """快速性能检查"""
        print("⚡ 快速性能检查...")
        
        # 小规模测试
        query = "人工智能应用"
        documents = create_sample_documents(10)
        
        start_time = time.time()
        result = self.reranker.rerank(query, documents)
        processing_time = time.time() - start_time
        
        cost = self.reranker.estimate_cost(len(documents))
        
        return {
            "status": "ok",
            "processing_time": processing_time,
            "documents_processed": len(documents),
            "estimated_cost": cost["estimated_cost_cny"],
            "throughput": len(documents) / processing_time,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """主函数"""
    benchmark = PerformanceBenchmark()
    benchmark.run_complete_benchmark()

if __name__ == "__main__":
    main()
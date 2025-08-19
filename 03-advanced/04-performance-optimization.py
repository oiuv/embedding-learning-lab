#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级功能第4课：性能优化系统
=========================

文本嵌入系统的性能优化，包括向量索引优化、近似最近邻搜索、缓存策略、分布式部署等。

学习目标：
1. 向量索引优化技术（FAISS、Pinecone）
2. 近似最近邻搜索算法
3. 缓存策略和内存优化
4. 批量处理优化
5. 分布式部署方案
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import pickle
import json
from datetime import datetime, timedelta
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
from functools import lru_cache
import psutil
import gc

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.embedding_client import EmbeddingClient

# 尝试导入FAISS（可选）
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS未安装，将使用numpy实现")

class PerformanceOptimizer:
    """性能优化系统"""
    
    def __init__(self):
        """初始化性能优化器"""
        self.client = EmbeddingClient()
        self.cache = {}
        self.index_cache = {}
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_queries': 0,
            'total_time': 0
        }
        
        # 初始化Redis缓存（如果可用）
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            print("✅ Redis缓存已连接")
        except:
            self.use_redis = False
            print("⚠️ Redis未连接，使用内存缓存")
        
        # FAISS索引
        self.faiss_index = None
        self.faiss_embeddings = None
        self.faiss_texts = []
    
    def load_benchmark_data(self, n_samples: int = 1000) -> List[str]:
        """加载基准测试数据"""
        base_texts = [
            "人工智能技术正在快速发展，机器学习算法不断优化",
            "深度学习在图像识别和语音处理领域取得重大突破",
            "自然语言处理技术让计算机能够理解和生成人类语言",
            "云计算技术提供了弹性的计算资源和存储能力",
            "区块链技术通过分布式账本保证数据的安全性和透明性",
            "物联网设备连接数量持续增长，智能家居应用普及",
            "大数据分析帮助企业从海量数据中提取有价值的信息",
            "推荐系统根据用户行为提供个性化内容推荐",
            "搜索引擎优化技术提升网站在搜索结果中的排名",
            "网络安全威胁日益严重，防护措施需要不断加强"
        ]
        
        # 生成更多变体
        texts = []
        for i in range(n_samples):
            base = base_texts[i % len(base_texts)]
            # 添加一些变化
            variation = f"{base} - 版本{i+1}"
            texts.append(variation)
        
        return texts
    
    def benchmark_embedding_generation(self, texts: List[str]) -> Dict:
        """基准测试嵌入生成性能"""
        print("🎯 基准测试嵌入生成性能...")
        
        results = {
            'total_texts': len(texts),
            'batch_sizes': [1, 5, 10, 20, 50],
            'results': {}
        }
        
        for batch_size in results['batch_sizes']:
            print(f"\n📊 测试批大小: {batch_size}")
            
            # 测试批处理性能
            start_time = time.time()
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = [self.client.get_embedding(text) for text in batch]
                embeddings.extend(batch_embeddings)
            
            elapsed_time = time.time() - start_time
            
            results['results'][batch_size] = {
                'total_time': elapsed_time,
                'time_per_text': elapsed_time / len(texts),
                'texts_per_second': len(texts) / elapsed_time,
                'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024  # MB
            }
            
            print(f"   总时间: {elapsed_time:.2f}s")
            print(f"   每文本时间: {results['results'][batch_size]['time_per_text']:.3f}s")
            print(f"   处理速度: {results['results'][batch_size]['texts_per_second']:.1f} texts/s")
            print(f"   内存使用: {results['results'][batch_size]['memory_usage']:.1f}MB")
        
        return results
    
    def build_faiss_index(self, texts: List[str]) -> Dict:
        """构建FAISS向量索引"""
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS不可用，跳过索引构建")
            return {'status': 'FAISS不可用'}
        
        print("🎯 构建FAISS向量索引...")
        
        start_time = time.time()
        
        # 获取所有文本的嵌入
        embeddings = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings).astype('float32')
        
        # 构建FAISS索引
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度
        self.faiss_index.add(embeddings_array)
        self.faiss_embeddings = embeddings_array
        self.faiss_texts = texts
        
        build_time = time.time() - start_time
        
        return {
            'status': '成功',
            'total_vectors': len(texts),
            'dimension': dimension,
            'build_time': build_time,
            'memory_usage': embeddings_array.nbytes / 1024 / 1024  # MB
        }
    
    def faiss_similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """使用FAISS进行相似度搜索"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            print("⚠️ FAISS索引不可用，使用numpy实现")
            return self.numpy_similarity_search(query, k)
        
        start_time = time.time()
        
        # 获取查询嵌入
        query_embedding = np.array(self.client.get_embedding(query)).astype('float32').reshape(1, -1)
        
        # 搜索相似文本
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.faiss_texts):
                results.append({
                    'text': self.faiss_texts[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        search_time = time.time() - start_time
        
        return {
            'results': results,
            'search_time': search_time,
            'results_per_second': 1 / search_time
        }
    
    def numpy_similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """使用numpy进行相似度搜索"""
        if not hasattr(self, 'numpy_embeddings'):
            print("请先构建numpy索引")
            return []
        
        start_time = time.time()
        
        # 获取查询嵌入
        query_embedding = np.array(self.client.get_embedding(query))
        
        # 计算相似度
        similarities = []
        for i, embedding in enumerate(self.numpy_embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((similarity, i))
        
        # 排序并返回前k个
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for score, idx in top_k:
            results.append({
                'text': self.numpy_texts[idx],
                'score': float(score),
                'index': int(idx)
            })
        
        search_time = time.time() - start_time
        
        return {
            'results': results,
            'search_time': search_time,
            'results_per_second': 1 / search_time
        }
    
    def build_numpy_index(self, texts: List[str]) -> Dict:
        """构建numpy索引"""
        print("🎯 构建numpy索引...")
        
        start_time = time.time()
        
        embeddings = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            embeddings.append(embedding)
        
        self.numpy_embeddings = np.array(embeddings)
        self.numpy_texts = texts
        
        build_time = time.time() - start_time
        
        return {
            'total_vectors': len(texts),
            'dimension': self.numpy_embeddings.shape[1],
            'build_time': build_time,
            'memory_usage': self.numpy_embeddings.nbytes / 1024 / 1024  # MB
        }
    
    def implement_caching_strategy(self, texts: List[str]) -> Dict:
        """实现缓存策略"""
        print("🎯 实现缓存策略...")
        
        # 多层缓存策略
        cache_stats = {
            'memory_cache': {'hits': 0, 'misses': 0},
            'redis_cache': {'hits': 0, 'misses': 0},
            'api_calls': 0
        }
        
        def get_embedding_with_cache(text: str) -> np.ndarray:
            """带缓存的嵌入获取"""
            cache_key = hashlib.md5(text.encode()).hexdigest()
            
            # 1. 检查内存缓存
            if cache_key in self.cache:
                cache_stats['memory_cache']['hits'] += 1
                return self.cache[cache_key]
            
            cache_stats['memory_cache']['misses'] += 1
            
            # 2. 检查Redis缓存
            if self.use_redis:
                try:
                    cached_embedding = self.redis_client.get(f"embedding:{cache_key}")
                    if cached_embedding:
                        embedding = json.loads(cached_embedding)
                        self.cache[cache_key] = embedding
                        cache_stats['redis_cache']['hits'] += 1
                        return embedding
                    cache_stats['redis_cache']['misses'] += 1
                except:
                    pass
            
            # 3. 调用API获取
            embedding = self.client.get_embedding(text)
            cache_stats['api_calls'] += 1
            
            # 保存到缓存
            self.cache[cache_key] = embedding
            if self.use_redis:
                try:
                    self.redis_client.setex(
                        f"embedding:{cache_key}", 
                        timedelta(hours=24), 
                        json.dumps(embedding)
                    )
                except:
                    pass
            
            return embedding
        
        # 测试缓存性能
        start_time = time.time()
        
        for text in texts:
            get_embedding_with_cache(text)
        
        total_time = time.time() - start_time
        
        return {
            'cache_stats': cache_stats,
            'total_time': total_time,
            'cache_hit_rate': (cache_stats['memory_cache']['hits'] + cache_stats['redis_cache']['hits']) / len(texts),
            'memory_cache_size': len(self.cache)
        }
    
    def parallel_processing_demo(self, texts: List[str], n_workers: int = 4) -> Dict:
        """并行处理演示"""
        print(f"🎯 并行处理演示（{n_workers}个工作进程）...")
        
        def process_batch(batch_texts: List[str]) -> List[np.ndarray]:
            """处理一批文本"""
            embeddings = []
            for text in batch_texts:
                embedding = self.client.get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        
        # 分割数据
        batch_size = max(1, len(texts) // n_workers)
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # 串行处理
        start_time = time.time()
        serial_embeddings = []
        for text in texts:
            serial_embeddings.append(self.client.get_embedding(text))
        serial_time = time.time() - start_time
        
        # 并行处理
        start_time = time.time()
        parallel_embeddings = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                batch_embeddings = future.result()
                parallel_embeddings.extend(batch_embeddings)
        
        parallel_time = time.time() - start_time
        
        return {
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': serial_time / parallel_time,
            'efficiency': (serial_time / parallel_time) / n_workers,
            'total_texts': len(texts),
            'batches': len(batches)
        }
    
    def memory_optimization_demo(self, texts: List[str]) -> Dict:
        """内存优化演示"""
        print("🎯 内存优化演示...")
        
        # 原始方法
        start_time = time.time()
        original_embeddings = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            original_embeddings.append(np.array(embedding, dtype=np.float32))
        
        original_memory = sum([emb.nbytes for emb in original_embeddings]) / 1024 / 1024
        original_time = time.time() - start_time
        
        # 优化方法：使用numpy数组
        start_time = time.time()
        optimized_embeddings = []
        for text in texts:
            embedding = self.client.get_embedding(text)
            optimized_embeddings.append(embedding)
        
        # 转换为单个numpy数组
        optimized_array = np.array(optimized_embeddings, dtype=np.float32)
        optimized_memory = optimized_array.nbytes / 1024 / 1024
        optimized_time = time.time() - start_time
        
        # 内存压缩
        compressed = np.round(optimized_array, decimals=4)
        compressed_memory = compressed.nbytes / 1024 / 1024
        
        return {
            'original_memory_mb': original_memory,
            'original_time': original_time,
            'optimized_memory_mb': optimized_memory,
            'optimized_time': optimized_time,
            'memory_reduction': (original_memory - optimized_memory) / original_memory,
            'compressed_memory_mb': compressed_memory,
            'compression_ratio': (original_memory - compressed_memory) / original_memory
        }
    
    def comprehensive_performance_test(self, n_samples: int = 100) -> Dict:
        """综合性能测试"""
        print("🎯 综合性能测试...")
        
        # 加载测试数据
        texts = self.load_benchmark_data(n_samples)
        
        # 1. 基准测试
        benchmark_results = self.benchmark_embedding_generation(texts[:20])
        
        # 2. 索引构建
        numpy_results = self.build_numpy_index(texts)
        
        # 3. 相似度搜索测试
        search_queries = ["人工智能技术应用", "体育赛事报道", "财经市场分析"]
        search_results = []
        
        for query in search_queries:
            result = self.numpy_similarity_search(query, k=5)
            search_results.append({
                'query': query,
                **result
            })
        
        # 4. 缓存测试
        cache_results = self.implement_caching_strategy(texts[:10])
        
        # 5. 并行处理测试
        parallel_results = self.parallel_processing_demo(texts[:20], n_workers=2)
        
        # 6. 内存优化测试
        memory_results = self.memory_optimization_demo(texts[:20])
        
        # 汇总结果
        summary = {
            'test_config': {
                'total_texts': n_samples,
                'test_queries': len(search_queries)
            },
            'benchmark': benchmark_results,
            'index_build': numpy_results,
            'search_performance': search_results,
            'cache_performance': cache_results,
            'parallel_performance': parallel_results,
            'memory_optimization': memory_results
        }
        
        return summary
    
    def generate_optimization_report(self, results: Dict) -> str:
        """生成优化报告"""
        report = f"""
# 性能优化报告

## 测试配置
- 测试样本数: {results['test_config']['total_texts']}
- 测试查询数: {results['test_config']['test_queries']}

## 嵌入生成性能
"""
        
        for batch_size, data in results['benchmark']['results'].items():
            report += f"""
### 批大小 {batch_size}
- 处理速度: {data['texts_per_second']:.1f} texts/s
- 每文本时间: {data['time_per_text']:.3f}s
- 内存使用: {data['memory_usage']:.1f}MB
"""
        
        report += f"""
## 索引构建
- 构建时间: {results['index_build']['build_time']:.2f}s
- 内存使用: {results['index_build']['memory_usage']:.1f}MB
- 向量维度: {results['index_build']['dimension']}

## 搜索性能
"""
        
        for search_result in results['search_performance']:
            report += f"""
### 查询: {search_result['query']}
- 搜索时间: {search_result['search_time']:.3f}s
- 处理速度: {search_result['results_per_second']:.1f} queries/s
"""
        
        report += f"""
## 缓存性能
- 缓存命中率: {results['cache_performance']['cache_hit_rate']:.2%}
- 内存缓存大小: {results['cache_performance']['memory_cache_size']}
- Redis可用: {self.use_redis}

## 并行处理
- 串行时间: {results['parallel_performance']['serial_time']:.2f}s
- 并行时间: {results['parallel_performance']['parallel_time']:.2f}s
- 加速比: {results['parallel_performance']['speedup']:.2f}x
- 效率: {results['parallel_performance']['efficiency']:.2%}

## 内存优化
- 内存减少: {results['memory_optimization']['memory_reduction']:.2%}
- 压缩比: {results['memory_optimization']['compression_ratio']:.2%}
"""
        
        return report
    
    def demo_performance_optimization(self):
        """演示性能优化系统"""
        print("🚀 高级功能第4课：性能优化系统")
        print("=" * 60)
        
        # 运行综合性能测试
        results = self.comprehensive_performance_test(n_samples=50)
        
        # 生成并显示报告
        report = self.generate_optimization_report(results)
        print(report)
        
        # 保存报告
        with open("03-advanced/performance_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print("\n🎉 性能优化演示完成！")
        print("\n核心技术总结：")
        print("   • 向量索引优化")
        print("   • 近似最近邻搜索")
        print("   • 缓存策略")
        print("   • 并行处理")
        print("   • 内存优化")
        print("\n实际应用场景：")
        print("   • 高性能搜索引擎")
        print("   • 实时推荐系统")
        print("   • 大规模文本处理")
        print("   • 分布式部署")
        print("\n下一阶段：实战项目开发")

def main():
    """主函数"""
    print("🚀 性能优化系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        optimizer = PerformanceOptimizer()
        optimizer.demo_performance_optimization()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
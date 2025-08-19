#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本排序工具类
=============

提供文本排序模型的统一接口和工具函数，支持gte-rerank模型的集成。
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

import dashscope
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RerankDocument:
    """排序文档数据结构"""
    text: str
    doc_id: str = None
    score: float = 0.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RerankResult:
    """排序结果数据结构"""
    document: RerankDocument
    relevance_score: float
    original_rank: int
    new_rank: int
    rank_change: int

class TextReranker:
    """文本排序器类"""
    
    def __init__(self, 
                 model: str = "gte-rerank-v2",
                 api_key: str = None,
                 max_documents: int = 100,
                 cache_enabled: bool = True,
                 cache_ttl: int = 3600,
                 timeout: int = 30):
        """
        初始化文本排序器
        
        Args:
            model: 排序模型名称
            api_key: API密钥
            max_documents: 最大文档数量限制
            cache_enabled: 是否启用缓存
            cache_ttl: 缓存有效期（秒）
            timeout: API调用超时时间（秒）
        """
        self.model = model
        self.max_documents = max_documents
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        
        # 设置API密钥
        if api_key:
            dashscope.api_key = api_key
        elif os.getenv("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            raise ValueError("请提供DASHSCOPE_API_KEY")
        
        # 缓存管理
        self._cache = {}
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        logger.info(f"✅ 文本排序器初始化完成 - 模型: {model}")
    
    def _generate_cache_key(self, query: str, documents: List[str], **kwargs) -> str:
        """生成缓存键"""
        content = json.dumps({
            "query": query,
            "documents": documents,
            "kwargs": kwargs
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache(self, key: str) -> Optional[List[RerankResult]]:
        """从缓存获取结果"""
        if not self.cache_enabled:
            return None
            
        if key in self._cache:
            cached_data = self._cache[key]
            if datetime.now() < cached_data["expires"]:
                self._cache_stats["hits"] += 1
                return cached_data["results"]
            else:
                del self._cache[key]
                self._cache_stats["evictions"] += 1
        
        self._cache_stats["misses"] += 1
        return None
    
    def _set_cache(self, key: str, results: List[RerankResult]):
        """设置缓存"""
        if not self.cache_enabled:
            return
            
        # 清理过期缓存
        current_time = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items() 
            if current_time >= v["expires"]
        ]
        for k in expired_keys:
            del self._cache[k]
            self._cache_stats["evictions"] += 1
        
        # 添加新缓存
        self._cache[key] = {
            "results": results,
            "expires": current_time + timedelta(seconds=self.cache_ttl)
        }
    
    def rerank(self, 
               query: str, 
               documents: List[RerankDocument],
               top_n: Optional[int] = None,
               return_documents: bool = True) -> List[RerankResult]:
        """
        重新排序文档
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_n: 返回前n个结果，None表示返回所有
            return_documents: 是否返回文档内容
        
        Returns:
            排序后的结果列表
        """
        if not documents:
            return []
        
        # 限制文档数量
        if len(documents) > self.max_documents:
            logger.warning(f"文档数量超过限制({len(documents)} > {self.max_documents})，将进行截断")
            documents = documents[:self.max_documents]
        
        # 检查缓存
        doc_texts = [doc.text for doc in documents]
        cache_key = self._generate_cache_key(query, doc_texts, top_n=top_n)
        cached_results = self._get_cache(cache_key)
        
        if cached_results:
            logger.debug("使用缓存结果")
            return cached_results
        
        try:
            # 调用排序API
            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=top_n or len(documents),
                return_documents=return_documents
            )
            
            if response.status_code == 200:
                results = self._process_response(documents, response.output.results)
                self._set_cache(cache_key, results)
                return results
            else:
                logger.error(f"排序API调用失败: {response}")
                return []
                
        except Exception as e:
            logger.error(f"文本排序失败: {e}")
            return []
    
    def _process_response(self, 
                         original_documents: List[RerankDocument],
                         api_results: List[Dict]) -> List[RerankResult]:
        """处理API响应"""
        results = []
        
        # 创建原始文档映射
        doc_map = {i: doc for i, doc in enumerate(original_documents)}
        
        for api_result in api_results:
            index = api_result.index
            relevance_score = api_result.relevance_score
            
            if index in doc_map:
                document = doc_map[index]
                result = RerankResult(
                    document=document,
                    relevance_score=relevance_score,
                    original_rank=index,
                    new_rank=len(results),
                    rank_change=index - len(results)
                )
                results.append(result)
        
        return results
    
    def batch_rerank(self, 
                    queries: List[str], 
                    documents_list: List[List[RerankDocument]],
                    top_n: Optional[int] = None) -> List[List[RerankResult]]:
        """
        批量重新排序
        
        Args:
            queries: 查询列表
            documents_list: 文档列表的列表
            top_n: 返回前n个结果
        
        Returns:
            排序结果列表的列表
        """
        if len(queries) != len(documents_list):
            raise ValueError("查询和文档列表数量不匹配")
        
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_n=top_n)
            results.append(reranked)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "evictions": self._cache_stats["evictions"],
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": len(self._cache) + self._cache_stats["evictions"]
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        logger.info("✅ 缓存已清空")
    
    def estimate_cost(self, num_documents: int) -> Dict[str, float]:
        """估算API调用成本"""
        # gte-rerank-v2: 0.0008元/千输入Token
        cost_per_1k_tokens = 0.0008
        
        # 估算平均每个文档的Token数（保守估计）
        avg_tokens_per_doc = 200
        total_tokens = num_documents * avg_tokens_per_doc
        
        cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_tokens": total_tokens,
            "estimated_cost_cny": cost,
            "cost_per_1k_tokens": cost_per_1k_tokens
        }

class AdvancedReranker(TextReranker):
    """高级文本排序器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_config = {
            "semantic_weight": 0.7,
            "keyword_weight": 0.2,
            "popularity_weight": 0.1
        }
    
    def hybrid_rank(self, 
                   query: str,
                   documents: List[RerankDocument],
                   query_embedding: List[float] = None,
                   doc_embeddings: List[List[float]] = None) -> List[RerankResult]:
        """
        混合排序（结合多种信号）
        
        Args:
            query: 查询文本
            documents: 文档列表
            query_embedding: 查询嵌入向量（可选）
            doc_embeddings: 文档嵌入向量列表（可选）
        
        Returns:
            排序结果
        """
        # 获取文本排序结果
        rerank_results = self.rerank(query, documents)
        
        if not rerank_results or not query_embedding or not doc_embeddings:
            return rerank_results
        
        # 计算语义相似度
        semantic_scores = []
        for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings)):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            semantic_scores.append((i, similarity))
        
        # 创建分数映射
        rerank_scores = {r.original_rank: r.relevance_score for r in rerank_results}
        
        # 计算混合分数
        hybrid_results = []
        for result in rerank_results:
            original_idx = result.original_rank
            semantic_score = next(s for i, s in semantic_scores if i == original_idx)
            
            # 标准化分数
            rerank_norm = result.relevance_score
            semantic_norm = (semantic_score + 1) / 2  # 余弦相似度标准化
            
            # 计算最终分数
            final_score = (
                self.weight_config["semantic_weight"] * semantic_norm +
                self.weight_config["rerank_weight"] * rerank_norm
            )
            
            hybrid_result = RerankResult(
                document=result.document,
                relevance_score=final_score,
                original_rank=result.original_rank,
                new_rank=result.new_rank,
                rank_change=result.rank_change
            )
            hybrid_results.append(hybrid_result)
        
        # 按最终分数排序
        hybrid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 更新排名
        for i, result in enumerate(hybrid_results):
            result.new_rank = i
        
        return hybrid_results

# 工具函数
def create_sample_documents(count: int = 10, category: str = "tech") -> List[RerankDocument]:
    """创建示例文档"""
    sample_tech_docs = [
        "人工智能技术在医疗诊断中的应用越来越广泛，特别是在影像识别和疾病预测方面",
        "深度学习算法在图像识别领域取得了突破性进展，准确率已经超过人类水平",
        "区块链技术为金融行业带来了革命性变化，特别是在支付和清算系统方面",
        "量子计算的发展将彻底改变密码学和计算科学的面貌",
        "物联网技术正在连接数十亿设备，构建智能化的生活和工作环境",
        "机器学习在金融风险评估中的应用显著提高了风险预测的准确性",
        "自然语言处理技术使得计算机能够理解和生成人类语言",
        "计算机视觉技术在自动驾驶汽车中的应用日益成熟",
        "大数据分析帮助企业从海量数据中提取有价值的商业洞察",
        "云计算技术为企业提供了灵活可扩展的计算资源"
    ]
    
    return [
        RerankDocument(
            text=text,
            doc_id=f"doc_{i}",
            score=np.random.uniform(0.3, 0.9),
            metadata={"category": category, "index": i}
        )
        for i, text in enumerate(sample_tech_docs[:count])
    ]

def benchmark_rerank_performance(reranker: TextReranker, 
                               test_cases: List[Tuple[str, List[RerankDocument]]]) -> Dict:
    """基准测试排序性能"""
    results = {
        "total_queries": len(test_cases),
        "total_documents": sum(len(docs) for _, docs in test_cases),
        "timing": {},
        "cache_stats": {}
    }
    
    start_time = time.time()
    
    for i, (query, documents) in enumerate(test_cases):
        case_start = time.time()
        reranked = reranker.rerank(query, documents)
        case_time = time.time() - case_start
        
        results["timing"][f"query_{i}"] = {
            "query": query,
            "document_count": len(documents),
            "time_seconds": case_time,
            "results_count": len(reranked)
        }
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["average_time_per_query"] = total_time / len(test_cases)
    results["cache_stats"] = reranker.get_cache_stats()
    
    return results

# 使用示例
if __name__ == "__main__":
    # 初始化排序器
    reranker = TextReranker()
    
    # 创建示例文档
    documents = create_sample_documents(5)
    query = "人工智能在医疗领域的应用"
    
    # 执行排序
    results = reranker.rerank(query, documents)
    
    print("🎯 文本排序结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. 分数: {result.relevance_score:.3f} - {result.document.text[:60]}...")
    
    # 显示缓存统计
    stats = reranker.get_cache_stats()
    print(f"\n📊 缓存统计: {stats}")
    
    # 估算成本
    cost = reranker.estimate_cost(len(documents))
    print(f"\n💰 成本估算: {cost}")
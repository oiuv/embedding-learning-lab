#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级教程第6课：文本排序模型系统集成
================================

本课程将教你如何将文本排序模型集成到现有的语义搜索和问答系统中。

学习目标：
1. 设计排序模型集成架构
2. 实现混合排序策略
3. 优化系统性能
4. 处理大规模数据排序
5. 添加排序结果缓存机制

"""

import os
import sys
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import time

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

# 初始化DashScope
import dashscope
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

@dataclass
class SearchDocument:
    """搜索文档"""
    doc_id: str
    title: str
    content: str
    score: float
    metadata: Dict
    embedding: List[float] = None

@dataclass
class RankingResult:
    """排序结果"""
    document: SearchDocument
    original_score: float
    rerank_score: float
    final_score: float
    rank_change: int

class TextRerankIntegrator:
    """文本排序集成器"""
    
    def __init__(self, cache_enabled: bool = True, cache_ttl: int = 3600):
        """初始化集成器"""
        self.client = EmbeddingClient()
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # 配置参数
        self.config = {
            "max_candidates": 100,      # 最大候选文档数
            "min_score_threshold": 0.1,  # 最小分数阈值
            "rerank_weight": 0.7,       # 排序权重
            "original_weight": 0.3,     # 原始分数权重
            "batch_size": 25            # 批量处理大小
        }
        
        print("✅ 文本排序集成器初始化完成")
    
    def _generate_cache_key(self, query: str, documents: List[str]) -> str:
        """生成缓存键"""
        content = f"{query}_{''.join(sorted(documents))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[RankingResult]]:
        """从缓存获取结果"""
        if not self.cache_enabled:
            return None
            
        if key in self.cache:
            cached_data = self.cache[key]
            if datetime.now() < cached_data["expires"]:
                self.cache_stats["hits"] += 1
                return cached_data["results"]
            else:
                del self.cache[key]
        
        self.cache_stats["misses"] += 1
        return None
    
    def _save_to_cache(self, key: str, results: List[RankingResult]):
        """保存结果到缓存"""
        if not self.cache_enabled:
            return
            
        self.cache[key] = {
            "results": results,
            "expires": datetime.now() + timedelta(seconds=self.cache_ttl)
        }
    
    def rerank_documents(self, 
                        query: str, 
                        documents: List[SearchDocument],
                        strategy: str = "hybrid") -> List[RankingResult]:
        """
        重新排序文档
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            strategy: 排序策略 ('embedding', 'rerank', 'hybrid')
        
        Returns:
            排序后的结果列表
        """
        if not documents:
            return []
        
        # 限制候选文档数量
        documents = documents[:self.config["max_candidates"]]
        
        # 检查缓存
        doc_texts = [f"{doc.title} {doc.content}" for doc in documents]
        cache_key = self._generate_cache_key(query, doc_texts)
        cached_results = self._get_from_cache(cache_key)
        
        if cached_results:
            return cached_results
        
        # 根据策略选择排序方法
        if strategy == "embedding":
            results = self._rank_by_embedding(query, documents)
        elif strategy == "rerank":
            results = self._rank_by_rerank(query, documents)
        elif strategy == "hybrid":
            results = self._rank_by_hybrid(query, documents)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 保存到缓存
        self._save_to_cache(cache_key, results)
        
        return results
    
    def _rank_by_embedding(self, query: str, documents: List[SearchDocument]) -> List[RankingResult]:
        """使用嵌入模型排序"""
        query_embedding = self.client.get_embedding(query)
        
        results = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.client.get_embedding(f"{doc.title} {doc.content}")
            
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            
            results.append(RankingResult(
                document=doc,
                original_score=doc.score,
                rerank_score=similarity,
                final_score=similarity,
                rank_change=0
            ))
        
        return sorted(results, key=lambda x: x.final_score, reverse=True)
    
    def _rank_by_rerank(self, query: str, documents: List[SearchDocument]) -> List[RankingResult]:
        """使用文本排序模型排序"""
        doc_texts = [f"{doc.title} {doc.content}" for doc in documents]
        
        try:
            response = dashscope.TextReRank.call(
                model="gte-rerank-v2",
                query=query,
                documents=doc_texts,
                top_n=len(documents)
            )
            
            if response.status_code == 200:
                results = []
                rerank_results = response.output.results
                
                # 创建原始排名映射
                original_ranks = {doc.doc_id: i for i, doc in enumerate(documents)}
                
                for rank, result in enumerate(rerank_results):
                    doc_idx = result.index
                    doc = documents[doc_idx]
                    
                    results.append(RankingResult(
                        document=doc,
                        original_score=doc.score,
                        rerank_score=result.relevance_score,
                        final_score=result.relevance_score,
                        rank_change=original_ranks[doc.doc_id] - rank
                    ))
                
                return results
            else:
                print(f"❌ 排序失败: {response}")
                return []
                
        except Exception as e:
            print(f"❌ 调用排序模型失败: {e}")
            return []
    
    def _rank_by_hybrid(self, query: str, documents: List[SearchDocument]) -> List[RankingResult]:
        """混合排序策略"""
        # 获取嵌入模型分数
        embedding_results = self._rank_by_embedding(query, documents)
        
        # 获取排序模型分数
        rerank_results = self._rank_by_rerank(query, documents)
        
        if not rerank_results:
            return embedding_results
        
        # 创建映射
        embedding_scores = {r.document.doc_id: r.rerank_score for r in embedding_results}
        rerank_scores = {r.document.doc_id: r.rerank_score for r in rerank_results}
        
        # 计算混合分数
        results = []
        for doc in documents:
            doc_id = doc.doc_id
            
            # 标准化分数
            embedding_norm = (embedding_scores[doc_id] + 1) / 2  # 余弦相似度标准化
            rerank_norm = rerank_scores.get(doc_id, 0)
            
            # 混合分数
            final_score = (
                self.config["original_weight"] * embedding_norm +
                self.config["rerank_weight"] * rerank_norm
            )
            
            # 计算排名变化
            original_rank = next(i for i, r in enumerate(embedding_results) 
                             if r.document.doc_id == doc_id)
            rerank_rank = next(i for i, r in enumerate(rerank_results) 
                           if r.document.doc_id == doc_id)
            
            results.append(RankingResult(
                document=doc,
                original_score=embedding_scores[doc_id],
                rerank_score=rerank_norm,
                final_score=final_score,
                rank_change=original_rank - rerank_rank
            ))
        
        return sorted(results, key=lambda x: x.final_score, reverse=True)
    
    def batch_rerank(self, 
                    queries: List[str], 
                    documents_list: List[List[SearchDocument]]) -> List[List[RankingResult]]:
        """批量重新排序"""
        results = []
        
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank_documents(query, documents)
            results.append(reranked)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        print("✅ 缓存已清空")

class ComplexDataGenerator:
    """复杂数据生成器"""
    
    def __init__(self):
        """初始化数据生成器"""
        self.seed_data = {
            "tech": {
                "titles": [
                    "人工智能在医疗诊断中的应用",
                    "深度学习算法的最新进展",
                    "区块链技术在金融领域的应用",
                    "量子计算的发展前景",
                    "物联网技术的标准化进程"
                ],
                "content_templates": [
                    "{topic}技术近年来发展迅速，特别是在{application}领域展现出巨大潜力。",
                    "研究表明，{topic}技术在解决{problem}问题时具有显著优势。",
                    "{topic}技术的商业化应用正在加速，预计将在{timeframe}内实现重大突破。"
                ]
            },
            "medical": {
                "titles": [
                    "肺癌早期筛查的新方法",
                    "糖尿病个性化治疗方案",
                    "心血管疾病预防策略",
                    "神经系统疾病的诊断技术",
                    "癌症免疫治疗的最新进展"
                ],
                "content_templates": [
                    "{disease}的早期诊断对于提高治愈率至关重要，{method}技术显示出良好前景。",
                    "{treatment}方法在{condition}治疗中效果显著，患者生存率提高了{percentage}%。",
                    "{technology}技术的引入使得{disease}的诊断准确率提升了{improvement}。"
                ]
            },
            "finance": {
                "titles": [
                    "数字货币的监管政策",
                    "金融科技对传统银行业的影响",
                    "量化投资策略的优化",
                    "区块链技术在支付系统中的应用",
                    "金融风险管理的创新方法"
                ],
                "content_templates": [
                    "{topic}在金融领域的应用正在改变传统的{business}模式。",
                    "{technology}技术的采用使得{process}效率提高了{percentage}%。",
                    "{innovation}为{sector}行业带来了新的发展机遇和挑战。"
                ]
            }
        }
    
    def generate_document_corpus(self, 
                                category: str, 
                                count: int,
                                include_noise: bool = True) -> List[SearchDocument]:
        """生成文档语料"""
        documents = []
        
        if category not in self.seed_data:
            raise ValueError(f"Unknown category: {category}")
        
        data = self.seed_data[category]
        
        for i in range(count):
            # 生成相关文档
            title = np.random.choice(data["titles"])
            template = np.random.choice(data["content_templates"])
            
            # 填充模板
            content = template.format(
                topic=title.split("在")[0] if "在" in title else title,
                application=f"{category}领域",
                problem="复杂问题",
                timeframe="未来5年",
                disease="疾病",
                method="新技术",
                treatment="新疗法",
                percentage="20-30%",
                technology="先进",
                improvement="显著",
                business="业务",
                process="流程",
                innovation="创新",
                sector=category
            )
            
            # 添加更多内容
            content += " " + self._generate_additional_content(category)
            
            score = np.random.uniform(0.3, 0.9)
            
            doc = SearchDocument(
                doc_id=f"{category}_{i:04d}",
                title=title,
                content=content,
                score=score,
                metadata={
                    "category": category,
                    "length": len(content),
                    "created": datetime.now().isoformat(),
                    "relevance": np.random.choice(["high", "medium", "low"])
                }
            )
            
            documents.append(doc)
        
        # 添加噪声文档（不相关文档）
        if include_noise:
            noise_count = max(1, count // 5)
            noise_docs = self._generate_noise_documents(noise_count)
            documents.extend(noise_docs)
        
        return documents
    
    def _generate_additional_content(self, category: str) -> str:
        """生成附加内容"""
        additional_sentences = {
            "tech": [
                "该技术的关键突破在于算法的优化和计算效率的提升。",
                "实际应用中需要考虑数据隐私和安全问题。",
                "未来的发展方向包括跨领域融合和标准化建设。"
            ],
            "medical": [
                "临床试验数据显示该方法具有良好的安全性和有效性。",
                "患者依从性和长期随访是研究成功的关键因素。",
                "个性化医疗方案需要根据患者具体情况进行调整。"
            ],
            "finance": [
                "监管环境的完善对于行业的健康发展至关重要。",
                "风险控制机制是系统稳定运行的重要保障。",
                "市场接受度和用户教育需要持续推进。"
            ]
        }
        
        return " ".join(np.random.choice(additional_sentences[category], 2))
    
    def _generate_noise_documents(self, count: int) -> List[SearchDocument]:
        """生成噪声文档"""
        noise_topics = [
            "宠物饲养技巧",
            "旅行攻略分享",
            "美食制作教程",
            "运动健身指南",
            "园艺种植知识"
        ]
        
        documents = []
        for i in range(count):
            topic = np.random.choice(noise_topics)
            content = f"这是一个关于{topic}的文档，内容完全不相关。"
            
            doc = SearchDocument(
                doc_id=f"noise_{i:04d}",
                title=f"噪声文档：{topic}",
                content=content,
                score=np.random.uniform(0.1, 0.3),
                metadata={
                    "category": "noise",
                    "length": len(content),
                    "created": datetime.now().isoformat(),
                    "relevance": "none"
                }
            )
            
            documents.append(doc)
        
        return documents

class IntegrationDemo:
    """集成演示类"""
    
    def __init__(self):
        """初始化演示"""
        self.reranker = TextRerankIntegrator()
        self.data_generator = ComplexDataGenerator()
        
        print("🚀 文本排序集成演示启动")
        print("=" * 50)
    
    def demo_search_integration(self):
        """搜索系统集成演示"""
        print("🔍 搜索系统集成演示")
        print("=" * 40)
        
        # 生成测试数据
        query = "人工智能在医疗领域的最新应用"
        documents = self.data_generator.generate_document_corpus("medical", 15)
        
        print(f"📊 生成 {len(documents)} 篇医疗相关文档")
        
        # 使用不同策略排序
        strategies = ["embedding", "rerank", "hybrid"]
        
        for strategy in strategies:
            print(f"\n🎯 使用{strategy}策略排序:")
            results = self.reranker.rerank_documents(query, documents, strategy=strategy)
            
            for i, result in enumerate(results[:5]):
                print(f"   {i+1}. {result.document.title[:50]}...")
                print(f"       原始分数: {result.original_score:.3f}")
                print(f"       重排分数: {result.rerank_score:.3f}")
                print(f"       最终分数: {result.final_score:.3f}")
                print(f"       排名变化: {result.rank_change:+.0f}")
    
    def demo_batch_processing(self):
        """批量处理演示"""
        print("\n⚡ 批量处理演示")
        print("=" * 40)
        
        # 生成多个查询的测试数据
        queries = [
            "人工智能在医疗诊断中的应用",
            "区块链技术在金融领域的应用",
            "深度学习算法的最新进展"
        ]
        
        documents_list = [
            self.data_generator.generate_document_corpus("medical", 10),
            self.data_generator.generate_document_corpus("finance", 10),
            self.data_generator.generate_document_corpus("tech", 10)
        ]
        
        start_time = time.time()
        results = self.reranker.batch_rerank(queries, documents_list)
        total_time = time.time() - start_time
        
        print(f"📊 批量处理完成!")
        print(f"   查询数量: {len(queries)}")
        print(f"   总文档数: {sum(len(docs) for docs in documents_list)}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均每个查询: {total_time/len(queries):.2f}s")
        
        for i, (query, docs) in enumerate(zip(queries, results)):
            print(f"\n   查询{i+1}: {query}")
            print(f"   前3个结果:")
            for j, result in enumerate(docs[:3]):
                print(f"      {j+1}. {result.document.title[:40]}... ({result.final_score:.3f})")
    
    def demo_cache_performance(self):
        """缓存性能演示"""
        print("\n💾 缓存性能演示")
        print("=" * 40)
        
        # 生成测试数据
        query = "人工智能技术发展趋势"
        documents = self.data_generator.generate_document_corpus("tech", 20)
        
        # 第一次调用（缓存未命中）
        start_time = time.time()
        results1 = self.reranker.rerank_documents(query, documents)
        first_time = time.time() - start_time
        
        # 第二次调用（缓存命中）
        start_time = time.time()
        results2 = self.reranker.rerank_documents(query, documents)
        second_time = time.time() - start_time
        
        # 检查缓存统计
        cache_stats = self.reranker.get_cache_stats()
        
        print(f"📊 缓存性能分析:")
        print(f"   首次调用时间: {first_time:.3f}s")
        print(f"   缓存调用时间: {second_time:.3f}s")
        print(f"   性能提升: {((first_time-second_time)/first_time*100):.1f}%")
        print(f"   缓存命中率: {cache_stats['hit_rate']:.2%}")
        print(f"   缓存大小: {cache_stats['cache_size']}")
    
    def run_integration_demo(self):
        """运行集成演示"""
        print("🎓 文本排序系统集成演示")
        print("=" * 60)
        
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
            return
        
        try:
            self.demo_search_integration()
            self.demo_batch_processing()
            self.demo_cache_performance()
            
            print("\n🎉 集成演示完成！")
            print("\n📚 你学会了：")
            print("✅ 文本排序模型的系统集成")
            print("✅ 混合排序策略的实现")
            print("✅ 缓存机制的优化")
            print("✅ 批量处理的效率提升")
            
        except Exception as e:
            print(f"❌ 演示运行失败: {e}")

def main():
    """主函数"""
    demo = IntegrationDemo()
    demo.run_integration_demo()

if __name__ == "__main__":
    main()
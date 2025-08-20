#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级课程第1课：语义搜索系统
============================

本课程将教你构建一个完整的语义搜索系统。

学习目标：
1. 理解语义搜索vs关键词搜索的区别
2. 构建文档索引系统
3. 实现智能搜索算法
4. 优化搜索结果排序
5. 添加搜索建议功能

"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI
import pickle
from datetime import datetime

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    text: str
    score: float
    metadata: Dict
    index: int

class SemanticSearchEngine:
    """语义搜索引擎类"""
    
    def __init__(self, api_key: str = None, index_file: str = "semantic_index.pkl"):
        """初始化搜索引擎"""
        try:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "text-embedding-v4"
            self.dimensions = 1024
            self.index_file = index_file
            self.documents = []
            self.embeddings = []
            self.metadata = []
            print("✅ 语义搜索引擎初始化成功！")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            sys.exit(1)
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ 获取嵌入失败: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def build_index(self, documents: List[Dict]) -> bool:
        """构建搜索索引"""
        print("🎯 构建搜索索引...")
        print("=" * 50)
        
        if not documents:
            print("❌ 文档列表为空")
            return False
        
        try:
            # 提取文本和元数据
            texts = [doc['text'] for doc in documents]
            self.metadata = [doc.get('metadata', {}) for doc in documents]
            
            # 获取嵌入
            print(f"📊 正在处理 {len(texts)} 个文档...")
            self.embeddings = self.get_embeddings_batch(texts)
            self.documents = texts
            
            if not self.embeddings:
                print("❌ 获取嵌入失败")
                return False
            
            # 保存索引
            self.save_index()
            
            print(f"✅ 索引构建完成！")
            print(f"   文档数量: {len(self.documents)}")
            print(f"   向量维度: {len(self.embeddings[0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 构建索引失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[SearchResult]:
        """执行语义搜索"""
        if not self.embeddings:
            print("❌ 搜索索引为空")
            return []
        
        try:
            # 获取查询嵌入
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # 计算相似度
            similarities = []
            for i, (embedding, doc) in enumerate(zip(self.embeddings, self.documents)):
                score = self.cosine_similarity(query_embedding, embedding)
                if score >= threshold:
                    similarities.append((i, score))
            
            # 排序并返回结果
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]
            
            results = []
            for idx, score in top_results:
                result = SearchResult(
                    text=self.documents[idx],
                    score=score,
                    metadata=self.metadata[idx],
                    index=idx
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def save_index(self):
        """保存索引到文件"""
        try:
            index_data = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.index_file, 'wb') as f:
                pickle.dump(index_data, f)
            
            print(f"✅ 索引已保存到 {self.index_file}")
            
        except Exception as e:
            print(f"❌ 保存索引失败: {e}")
    
    def load_index(self) -> bool:
        """从文件加载索引"""
        try:
            if not os.path.exists(self.index_file):
                print("⚠️ 索引文件不存在")
                return False
            
            with open(self.index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.embeddings = index_data['embeddings']
            self.metadata = index_data['metadata']
            
            print(f"✅ 索引加载成功！")
            print(f"   文档数量: {len(self.documents)}")
            print(f"   创建时间: {index_data['created_at']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载索引失败: {e}")
            return False
    
    def add_documents(self, new_documents: List[Dict]) -> bool:
        """添加新文档到索引"""
        try:
            if not new_documents:
                return True
            
            # 获取新文档的嵌入
            texts = [doc['text'] for doc in new_documents]
            new_embeddings = self.get_embeddings_batch(texts)
            new_metadata = [doc.get('metadata', {}) for doc in new_documents]
            
            if not new_embeddings:
                return False
            
            # 添加到现有索引
            self.documents.extend(texts)
            self.embeddings.extend(new_embeddings)
            self.metadata.extend(new_metadata)
            
            # 保存更新后的索引
            self.save_index()
            
            print(f"✅ 成功添加 {len(new_documents)} 个文档")
            return True
            
        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return False
    
    def get_search_suggestions(self, query: str, max_suggestions: int = 5) -> List[str]:
        """获取搜索建议"""
        if not self.documents:
            return []
        
        # 简单的搜索建议：基于文档文本的前缀匹配
        suggestions = []
        query_lower = query.lower()
        
        for doc in self.documents:
            if query_lower in doc.lower():
                # 提取匹配的短语
                words = doc.split()
                for i in range(len(words)):
                    phrase = ' '.join(words[i:i+3])  # 3-gram
                    if query_lower in phrase.lower() and phrase not in suggestions:
                        suggestions.append(phrase)
                        if len(suggestions) >= max_suggestions:
                            break
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions[:max_suggestions]

def demo_product_search():
    """演示产品搜索"""
    print("🎯 产品搜索演示")
    print("=" * 50)
    
    # 示例产品数据
    products = [
        {
            "text": "iPhone 15 Pro Max 256GB 原色钛金属 5G手机",
            "metadata": {
                "category": "手机",
                "brand": "Apple",
                "price": 9999,
                "rating": 4.8
            }
        },
        {
            "text": "华为Mate 60 Pro 12GB+512GB 雅黑 卫星通信",
            "metadata": {
                "category": "手机",
                "brand": "Huawei",
                "price": 6999,
                "rating": 4.7
            }
        },
        {
            "text": "小米14 Ultra 16GB+1TB 钛金属版 徕卡影像",
            "metadata": {
                "category": "手机",
                "brand": "Xiaomi",
                "price": 5999,
                "rating": 4.6
            }
        },
        {
            "text": "MacBook Pro M3 14英寸 18GB+512GB 深空黑",
            "metadata": {
                "category": "笔记本",
                "brand": "Apple",
                "price": 14999,
                "rating": 4.9
            }
        },
        {
            "text": "戴森V12无线吸尘器 手持除螨 激光探测",
            "metadata": {
                "category": "家电",
                "brand": "Dyson",
                "price": 3999,
                "rating": 4.5
            }
        },
        {
            "text": "索尼WH-1000XM5 降噪耳机 无线蓝牙",
            "metadata": {
                "category": "耳机",
                "brand": "Sony",
                "price": 2499,
                "rating": 4.7
            }
        }
    ]
    
    # 创建搜索引擎
    engine = SemanticSearchEngine()
    
    # 构建索引
    if engine.build_index(products):
        # 执行搜索
        queries = [
            "最好的拍照手机",
            "苹果的产品",
            "无线耳机",
            "办公用的笔记本电脑",
            "性价比高的手机"
        ]
        
        for query in queries:
            print(f"\n🔍 搜索: '{query}'")
            results = engine.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result.text}")
                    print(f"      相似度: {result.score:.3f}")
                    print(f"      品牌: {result.metadata['brand']}")
                    print(f"      价格: ¥{result.metadata['price']}")
                    print(f"      评分: {result.metadata['rating']}")
            else:
                print("   未找到相关结果")

def demo_document_search():
    """演示文档搜索"""
    print("\n🎯 文档搜索演示")
    print("=" * 50)
    
    # 示例文档数据
    documents = [
        {
            "text": "机器学习是人工智能的一个分支，专注于让计算机从数据中学习",
            "metadata": {
                "category": "技术",
                "tags": ["AI", "机器学习"],
                "author": "张教授",
                "date": "2024-01-15"
            }
        },
        {
            "text": "深度学习使用神经网络处理复杂问题，需要大量计算资源",
            "metadata": {
                "category": "技术",
                "tags": ["AI", "深度学习"],
                "author": "李博士",
                "date": "2024-01-20"
            }
        },
        {
            "text": "自然语言处理让计算机能够理解和处理人类语言",
            "metadata": {
                "category": "技术",
                "tags": ["AI", "NLP"],
                "author": "王研究员",
                "date": "2024-01-25"
            }
        },
        {
            "text": "Python是最流行的数据科学编程语言，有丰富的机器学习库",
            "metadata": {
                "category": "编程",
                "tags": ["Python", "数据科学"],
                "author": "陈工程师",
                "date": "2024-02-01"
            }
        },
        {
            "text": "计算机视觉可以识别和分析图像内容，应用于人脸识别和自动驾驶",
            "metadata": {
                "category": "技术",
                "tags": ["AI", "计算机视觉"],
                "author": "赵专家",
                "date": "2024-02-05"
            }
        }
    ]
    
    # 创建搜索引擎
    engine = SemanticSearchEngine(index_file="document_index.pkl")
    
    # 构建索引
    if engine.build_index(documents):
        # 执行文档搜索
        queries = [
            "什么是机器学习",
            "深度学习需要哪些资源",
            "Python在数据科学中的应用",
            "计算机视觉的实际应用",
            "自然语言处理技术"
        ]
        
        for query in queries:
            print(f"\n🔍 文档搜索: '{query}'")
            results = engine.search(query, top_k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result.text}")
                    print(f"      相似度: {result.score:.3f}")
                    print(f"      作者: {result.metadata['author']}")
                    print(f"      日期: {result.metadata['date']}")
            else:
                print("   未找到相关文档")

def demo_search_suggestions():
    """演示搜索建议"""
    print("\n🎯 搜索建议演示")
    print("=" * 50)
    
    # 使用产品搜索的索引
    products = [
        {"text": "iPhone 15 Pro Max 智能手机", "metadata": {}},
        {"text": "华为Mate 60 Pro 卫星通信手机", "metadata": {}},
        {"text": "小米14 Ultra 徕卡影像手机", "metadata": {}},
        {"text": "MacBook Pro M3 笔记本电脑", "metadata": {}},
        {"text": "戴森V12 无线吸尘器", "metadata": {}}
    ]
    
    engine = SemanticSearchEngine()
    engine.build_index(products)
    
    # 测试搜索建议
    test_queries = ["iPhone", "华为", "小米", "笔记本", "吸尘"]
    
    for query in test_queries:
        suggestions = engine.get_search_suggestions(query)
        print(f"\n🔍 查询: '{query}'")
        print(f"   搜索建议: {suggestions}")

def main():
    """主函数"""
    print("🚀 中级课程第1课：语义搜索系统")
    print("=" * 60)
    print("本课程将教你构建一个完整的语义搜索系统。\n")
    
    try:
        # 检查API密钥
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("🔑 API密钥检查")
            print("-" * 30)
            print("⚠️ 未检测到 DASHSCOPE_API_KEY 环境变量")
            print("\n解决方法：")
            print("1. 临时设置: set DASHSCOPE_API_KEY=你的密钥 (Windows)")
            print("2. 临时设置: export DASHSCOPE_API_KEY=你的密钥 (Linux/Mac)")
            print("\n📝 获取API密钥：")
            print("   访问 https://dashscope.console.aliyun.com 申请")
            return
        else:
            print("✅ 检测到API密钥")
        
        input("\n🛒 按回车键开始产品搜索演示...")
        print("\n" + "="*60)
        demo_product_search()
        
        input("\n📄 按回车键开始文档搜索演示...")
        print("\n" + "="*60)
        demo_document_search()
        
        input("\n💡 按回车键查看搜索建议功能...")
        print("\n" + "="*60)
        demo_search_suggestions()
        
        print("\n" + "="*60)
        print("🎉 语义搜索课程完成！")
        print("🎯 你已经学会了：")
        print("✅ 构建搜索索引")
        print("✅ 实现语义搜索")
        print("✅ 搜索结果排序")
        print("✅ 添加搜索建议")
        print("✅ 索引持久化")
        print("\n📂 索引文件已保存为 .pkl 文件")
        print("\n🚀 准备进入下一课程...")
        print("\n中级模块：02-text-classification.py - 文本分类")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 课程已中断，欢迎下次继续学习！")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        print("🔄 请检查网络连接和API配置")
    finally:
        input("\n📚 按回车键退出课程...")

if __name__ == "__main__":
    main()
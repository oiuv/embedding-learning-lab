#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中级挑战1：多语言支持系统
======================

实现支持多语言的文本嵌入系统，处理中文、英文、日文等不同语言。

挑战目标：
1. 多语言文本预处理
2. 跨语言语义搜索
3. 语言检测功能
4. 多语言可视化
5. 语言特定优化
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple
import re
import langdetect
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

class MultilingualSupportSystem:
    """多语言支持系统"""
    
    def __init__(self):
        self.client = EmbeddingClient()
        self.language_patterns = {
            'chinese': re.compile(r'[\u4e00-\u9fff]+'),
            'japanese': re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]+'),
            'korean': re.compile(r'[\uac00-\ud7af]+'),
            'english': re.compile(r'[a-zA-Z]+'),
            'numbers': re.compile(r'\d+')
        }
    
    def load_multilingual_data(self) -> List[Dict[str, str]]:
        """加载多语言数据"""
        multilingual_texts = [
            {
                'text': '机器学习是人工智能的重要分支',
                'language': 'zh',
                'translation': 'Machine learning is an important branch of artificial intelligence'
            },
            {
                'text': 'Deep learning is revolutionizing AI technology',
                'language': 'en',
                'translation': '深度学习正在革命性地改变AI技术'
            },
            {
                'text': '機械学習は人工知能の重要な分野です',
                'language': 'ja',
                'translation': 'Machine learning is an important field of artificial intelligence'
            },
            {
                'text': 'Python编程语言在数据科学中广泛应用',
                'language': 'zh',
                'translation': 'Python programming language is widely used in data science'
            },
            {
                'text': 'Data science combines statistics and programming',
                'language': 'en',
                'translation': '数据科学结合了统计学和编程'
            },
            {
                'text': 'データサイエンスは統計学とプログラミングを組み合わせる',
                'language': 'ja',
                'translation': 'Data science combines statistics and programming'
            }
        ]
        return multilingual_texts
    
    def detect_language(self, text: str) -> str:
        """检测文本语言
        
        挑战：实现准确的语言检测
        """
        try:
            # 使用langdetect库
            detected = langdetect.detect(text)
            return detected
        except:
            # 备用检测方法
            if self.language_patterns['chinese'].search(text):
                return 'zh'
            elif self.language_patterns['japanese'].search(text):
                return 'ja'
            elif self.language_patterns['korean'].search(text):
                return 'ko'
            elif self.language_patterns['english'].search(text):
                return 'en'
            else:
                return 'unknown'
    
    def preprocess_multilingual_text(self, text: str, language: str) -> str:
        """多语言文本预处理
        
        挑战：针对不同语言进行适当的预处理
        """
        # 基础清理
        text = text.strip()
        
        if language == 'zh':
            # 中文处理
            # 移除标点符号，保留中文字符
            text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
        elif language == 'ja':
            # 日文处理
            # 保留日文汉字、平假名、片假名
            text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\w]', '', text)
        elif language == 'en':
            # 英文处理
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def cross_language_search(self, query: str, target_language: str, texts: List[Dict]) -> List[Dict]:
        """跨语言语义搜索
        
        挑战：实现跨语言的语义匹配
        """
        # 预处理查询
        query_language = self.detect_language(query)
        query_processed = self.preprocess_multilingual_text(query, query_language)
        
        # 获取查询嵌入
        query_embedding = self.client.get_embedding(query_processed)
        
        # 搜索目标语言文本
        results = []
        
        for text_data in texts:
            if text_data['language'] == target_language:
                # 获取目标文本嵌入
                target_text = self.preprocess_multilingual_text(
                    text_data['text'], 
                    text_data['language']
                )
                target_embedding = self.client.get_embedding(target_text)
                
                # 计算相似度
                similarity = np.dot(query_embedding, target_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(target_embedding)
                )
                
                results.append({
                    'text': text_data['text'],
                    'language': text_data['language'],
                    'translation': text_data['translation'],
                    'similarity': similarity,
                    'query_language': query_language
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def analyze_language_distribution(self, texts: List[str]) -> Dict[str, int]:
        """分析语言分布"""
        language_counts = Counter()
        
        for text in texts:
            lang = self.detect_language(text)
            language_counts[lang] += 1
        
        return dict(language_counts)
    
    def create_multilingual_embeddings(self, texts: List[Dict]) -> Dict[str, List[float]]:
        """创建多语言嵌入"""
        embeddings = {}
        
        for text_data in texts:
            language = text_data['language']
            text = text_data['text']
            
            # 预处理
            processed_text = self.preprocess_multilingual_text(text, language)
            
            # 获取嵌入
            embedding = self.client.get_embedding(processed_text)
            
            embeddings[text_data['text']] = {
                'embedding': embedding,
                'language': language,
                'processed_text': processed_text
            }
        
        return embeddings
    
    def multilingual_clustering(self, texts: List[Dict], n_clusters: int = 3) -> Dict:
        """多语言聚类分析"""
        # 获取所有文本的嵌入
        embeddings = []
        text_info = []
        
        for text_data in texts:
            processed_text = self.preprocess_multilingual_text(
                text_data['text'], 
                text_data['language']
            )
            embedding = self.client.get_embedding(processed_text)
            
            embeddings.append(embedding)
            text_info.append(text_data)
        
        embeddings_array = np.array(embeddings)
        
        # 使用K-means聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)
        
        # 构建聚类结果
        clusters = [[] for _ in range(n_clusters)]
        
        for i, (text_data, label) in enumerate(zip(text_info, labels)):
            clusters[label].append({
                'text': text_data['text'],
                'language': text_data['language'],
                'translation': text_data['translation'],
                'cluster': label
            })
        
        return {
            'clusters': clusters,
            'labels': labels,
            'embeddings': embeddings_array
        }
    
    def calculate_cross_language_similarity(self, text1: str, text2: str) -> float:
        """计算跨语言相似度"""
        # 检测语言
        lang1 = self.detect_language(text1)
        lang2 = self.detect_language(text2)
        
        # 预处理
        processed1 = self.preprocess_multilingual_text(text1, lang1)
        processed2 = self.preprocess_multilingual_text(text2, lang2)
        
        # 获取嵌入
        embedding1 = self.client.get_embedding(processed1)
        embedding2 = self.client.get_embedding(processed2)
        
        # 计算相似度
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return similarity
    
    def run_multilingual_challenge(self):
        """运行多语言挑战"""
        print("🌍 中级挑战1：多语言支持系统")
        print("=" * 60)
        
        # 加载多语言数据
        multilingual_data = self.load_multilingual_data()
        print(f"📊 加载 {len(multilingual_data)} 个多语言文本")
        
        # 语言检测
        print("\n🔍 语言检测演示")
        print("-" * 30)
        
        for text_data in multilingual_data[:3]:
            detected = self.detect_language(text_data['text'])
            print(f"   文本: {text_data['text'][:30]}...")
            print(f"   实际语言: {text_data['language']}")
            print(f"   检测语言: {detected}")
            print(f"   检测正确: {'✅' if detected == text_data['language'] else '❌'}")
            print()
        
        # 跨语言搜索
        print("🔍 跨语言语义搜索演示")
        print("-" * 30)
        
        queries = [
            {'query': 'machine learning', 'target_lang': 'zh'},
            {'query': '人工智能', 'target_lang': 'en'},
            {'query': 'データサイエンス', 'target_lang': 'zh'}
        ]
        
        for search_config in queries:
            results = self.cross_language_search(
                search_config['query'], 
                search_config['target_lang'], 
                multilingual_data
            )
            
            print(f"\n   查询: {search_config['query']} ({self.detect_language(search_config['query'])}) → {search_config['target_lang']}")
            for result in results[:2]:
                print(f"      {result['text']} (相似度: {result['similarity']:.3f})")
        
        # 语言分布分析
        print("\n📊 语言分布分析")
        print("-" * 30)
        
        all_texts = [item['text'] for item in multilingual_data]
        distribution = self.analyze_language_distribution(all_texts)
        
        for lang, count in distribution.items():
            print(f"   {lang}: {count} 个文本")
        
        # 多语言聚类
        print("\n🎯 多语言聚类分析")
        print("-" * 30)
        
        clustering_result = self.multilingual_clustering(multilingual_data)
        
        for i, cluster in enumerate(clustering_result['clusters']):
            print(f"\n   聚类 {i+1}:")
            for item in cluster:
                print(f"      [{item['language']}] {item['text']}")
        
        # 跨语言相似度测试
        print("\n🎯 跨语言相似度测试")
        print("-" * 30)
        
        test_pairs = [
            ("机器学习是人工智能的重要分支", "Machine learning is an important branch of AI"),
            ("深度学习", "Deep learning"),
            ("Python编程", "Python programming")
        ]
        
        for text1, text2 in test_pairs:
            similarity = self.calculate_cross_language_similarity(text1, text2)
            print(f"   相似度: {text1[:15]}... ↔ {text2[:15]}... = {similarity:.3f}")
        
        print("\n✅ 多语言挑战完成！")
        print("\n🎓 学习要点：")
        print("   • 语言检测：准确识别文本语言")
        print("   • 跨语言匹配：不同语言的语义对齐")
        print("   • 多语言聚类：语言无关的内容分组")
        print("   • 预处理策略：针对不同语言的优化")

def main():
    """主函数"""
    print("🌍 中级挑战：多语言支持系统")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        challenge = MultilingualSupportSystem()
        challenge.run_multilingual_challenge()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
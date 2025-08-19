#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战项目4：文档分析工具
==================

智能文档分析系统，实现文档分类、关键信息提取、相似文档发现、分析报告生成。

项目功能：
1. 文档自动分类
2. 关键信息提取
3. 相似文档发现
4. 内容摘要生成
5. 分析报告可视化
6. 批量文档处理

技术栈：
- 文档处理：Python-docx, PyPDF2
- 文本嵌入：text-embedding-v4
- 分类算法：多种机器学习算法
- 可视化：Matplotlib, Plotly
- 报告生成：Jinja2模板
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime
import hashlib
from dataclasses import dataclass
import re
from collections import Counter
import shutil
import mimetypes

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.embedding_client import EmbeddingClient

# 尝试导入文档处理库
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyMuPDF未安装，PDF处理功能受限")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️ python-docx未安装，Word处理功能受限")

@dataclass
class DocumentInfo:
    """文档信息"""
    doc_id: str
    filename: str
    file_path: str
    file_type: str
    size: int
    created_at: datetime
    content: str
    category: str
    keywords: List[str]
    summary: str
    embedding: List[float]

class DocumentAnalyzer:
    """文档分析工具"""
    
    def __init__(self, db_path: str = "document_analyzer.db"):
        """初始化文档分析器"""
        self.client = EmbeddingClient()
        self.db_path = db_path
        self.init_database()
        self.documents = {}
        self.supported_types = ['.txt', '.md', '.pdf', '.docx']
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文档表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT,
                size INTEGER,
                content TEXT,
                category TEXT,
                keywords TEXT,
                summary TEXT,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 文档分类表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 相似文档关系表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_similarity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc1_id TEXT,
                doc2_id TEXT,
                similarity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc1_id) REFERENCES documents(id),
                FOREIGN KEY (doc2_id) REFERENCES documents(id)
            )
        ''')
        
        # 分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                analysis_type TEXT,
                result_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_file(self, file_path: str) -> str:
        """从文件提取文本"""
        if not os.path.exists(file_path):
            return ""
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt' or file_extension == '.md':
            return self.extract_text_from_txt(file_path)
        elif file_extension == '.pdf' and PDF_AVAILABLE:
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx' and DOCX_AVAILABLE:
            return self.extract_text_from_docx(file_path)
        else:
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """从文本文件提取内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """从PDF提取文本"""
        if not PDF_AVAILABLE:
            return "PDF处理功能不可用"
        
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"PDF提取失败: {str(e)}"
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """从Word文档提取文本"""
        if not DOCX_AVAILABLE:
            return "Word处理功能不可用"
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Word提取失败: {str(e)}"
    
    def add_document(self, file_path: str, category: str = None) -> str:
        """添加文档"""
        if not os.path.exists(file_path):
            return None
        
        # 生成文档ID
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        # 提取文本
        content = self.extract_text_from_file(file_path)
        if not content:
            return None
        
        # 获取文件信息
        filename = os.path.basename(file_path)
        file_type = os.path.splitext(file_path)[1]
        file_size = os.path.getsize(file_path)
        
        # 生成分类
        if not category:
            category = self.auto_classify(content)
        
        # 提取关键词
        keywords = self.extract_keywords(content)
        
        # 生成摘要
        summary = self.generate_summary(content)
        
        # 生成嵌入
        embedding = self.client.get_embedding(content)
        embedding_str = json.dumps(embedding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents (id, filename, file_path, file_type, size, content, category, keywords, summary, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (doc_id, filename, file_path, file_type, file_size, content, category, json.dumps(keywords), summary, embedding_str))
        
        conn.commit()
        conn.close()
        
        # 更新内存缓存
        self.documents[doc_id] = DocumentInfo(
            doc_id=doc_id,
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            size=file_size,
            created_at=datetime.now(),
            content=content,
            category=category,
            keywords=keywords,
            summary=summary,
            embedding=embedding
        )
        
        return doc_id
    
    def auto_classify(self, content: str) -> str:
        """自动文档分类"""
        # 简单的关键词分类
        tech_keywords = ['机器学习', '深度学习', '人工智能', '编程', '算法', '数据', '代码', '开发']
        business_keywords = ['商业', '管理', '营销', '财务', '投资', '企业', '市场', '战略']
        education_keywords = ['教育', '学习', '教学', '课程', '培训', '学校', '学生', '知识']
        
        content_lower = content.lower()
        
        tech_score = sum(1 for kw in tech_keywords if kw in content_lower)
        business_score = sum(1 for kw in business_keywords if kw in content_lower)
        education_score = sum(1 for kw in education_keywords if kw in content_lower)
        
        scores = {'技术': tech_score, '商业': business_score, '教育': education_score}
        return max(scores, key=scores.get) or '其他'
    
    def extract_keywords(self, content: str, n_keywords: int = 10) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        words = re.findall(r'\w+', content.lower())
        
        # 过滤停用词
        stop_words = {'的', '了', '在', '是', '和', '与', '为', '对', '中', '上', '下', '这', '那', '有', '可以', '需要', '使用', '进行', '通过', '能够'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # 统计词频
        word_counts = Counter(filtered_words)
        
        # 返回高频词
        return [w for w, c in word_counts.most_common(n_keywords)]
    
    def generate_summary(self, content: str, max_length: int = 200) -> str:
        """生成文档摘要"""
        sentences = re.split(r'[。！？\.\!\?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return content[:max_length] + '...' if len(content) > max_length else content
        
        # 简单的摘要：选择前几个句子
        summary = ""
        for sentence in sentences[:3]:  # 最多3个句子
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + "。"
            else:
                break
        
        return summary or (content[:max_length] + '...' if len(content) > max_length else content)
    
    def find_similar_documents(self, doc_id: str, limit: int = 5) -> List[Dict]:
        """查找相似文档"""
        if doc_id not in self.documents:
            return []
        
        target_doc = self.documents[doc_id]
        target_embedding = np.array(target_doc.embedding)
        
        # 获取所有文档
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, content, category, keywords, embedding
            FROM documents WHERE id != ?
        ''', (doc_id,))
        
        docs = cursor.fetchall()
        conn.close()
        
        # 计算相似度
        similarities = []
        for doc_data in docs:
            doc_id2, filename, content, category, keywords_str, embedding_str = doc_data
            
            doc_embedding = np.array(json.loads(embedding_str))
            similarity = np.dot(target_embedding, doc_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(doc_embedding)
            )
            
            keywords = json.loads(keywords_str)
            
            similarities.append({
                'doc_id': doc_id2,
                'filename': filename,
                'title': filename,  # 使用文件名作为标题
                'content': content[:200] + '...' if len(content) > 200 else content,
                'category': category,
                'keywords': keywords,
                'similarity': similarity
            })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 保存相似关系
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sim in similarities[:limit]:
            cursor.execute('''
                INSERT OR REPLACE INTO document_similarity (doc1_id, doc2_id, similarity_score)
                VALUES (?, ?, ?)
            ''', (doc_id, sim['doc_id'], sim['similarity']))
        
        conn.commit()
        conn.close()
        
        return similarities[:limit]
    
    def batch_process_directory(self, directory: str, category: str = None) -> List[str]:
        """批量处理目录"""
        if not os.path.exists(directory):
            return []
        
        processed_docs = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in self.supported_types:
                    doc_id = self.add_document(file_path, category)
                    if doc_id:
                        processed_docs.append(doc_id)
        
        return processed_docs
    
    def generate_analysis_report(self, doc_id: str) -> Dict:
        """生成分析报告"""
        if doc_id not in self.documents:
            return {}
        
        doc = self.documents[doc_id]
        
        # 获取相似文档
        similar_docs = self.find_similar_documents(doc_id, limit=3)
        
        # 统计数据
        content_length = len(doc.content)
        word_count = len(re.findall(r'\w+', doc.content))
        
        # 分析结果
        report = {
            'document_info': {
                'doc_id': doc.doc_id,
                'filename': doc.filename,
                'file_type': doc.file_type,
                'size': doc.size,
                'category': doc.category,
                'keywords': doc.keywords,
                'summary': doc.summary
            },
            'statistics': {
                'content_length': content_length,
                'word_count': word_count,
                'keywords_count': len(doc.keywords)
            },
            'similar_documents': similar_docs,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # 保存分析结果
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results (doc_id, analysis_type, result_data)
            VALUES (?, ?, ?)
        ''', (doc_id, 'comprehensive_analysis', json.dumps(report)))
        
        conn.commit()
        conn.close()
        
        return report
    
    def load_sample_documents(self) -> List[str]:
        """加载示例文档"""
        sample_contents = [
            {
                'title': '机器学习基础教程',
                'content': '''
                机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。
                主要类型包括监督学习、无监督学习和强化学习。
                监督学习使用标记数据训练模型，无监督学习发现数据中的隐藏模式，
                强化学习通过奖励和惩罚机制优化决策。
                ''',
                'category': '技术',
                'tags': ['机器学习', 'AI', '教程']
            },
            {
                'title': 'Python编程入门',
                'content': '''
                Python是一种易学易用的编程语言，广泛应用于数据分析、机器学习、Web开发等领域。
                本教程涵盖Python基础语法、数据结构、函数、类等核心概念。
                通过实例学习如何编写高效、可维护的Python代码。
                ''',
                'category': '编程',
                'tags': ['Python', '编程', '入门']
            },
            {
                'title': '数据科学工作流程',
                'content': '''
                数据科学工作流程包括数据收集、数据清洗、数据探索、数据建模和结果解释。
                使用Python和相关库如Pandas、NumPy、Scikit-learn等工具进行数据分析。
                通过可视化技术展示数据洞察，支持业务决策。
                ''',
                'category': '数据分析',
                'tags': ['数据科学', '工作流程', 'Python']
            }
        ]
        
        doc_ids = []
        for i, content in enumerate(sample_contents):
            doc_id = f"sample_{i+1}"
            
            # 创建临时文件
            temp_file = f"temp_doc_{i+1}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content['content'])
            
            # 添加文档
            doc_id = self.add_document(temp_file, content['category'])
            if doc_id:
                doc_ids.append(doc_id)
            
            # 清理临时文件
            os.remove(temp_file)
        
        return doc_ids
    
    def demo_document_analyzer(self):
        """演示文档分析工具"""
        print("🚀 实战项目4：文档分析工具")
        print("=" * 60)
        
        # 创建示例文档目录
        sample_dir = "sample_documents"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        # 加载示例文档
        print("📚 创建示例文档...")
        doc_ids = self.load_sample_documents()
        
        print(f"✅ 已加载 {len(doc_ids)} 个示例文档")
        
        # 演示文档分析功能
        if doc_ids:
            print("\n🎯 文档分析演示")
            print("=" * 50)
            
            for doc_id in doc_ids:
                doc = self.documents[doc_id]
                
                print(f"\n📄 文档: {doc.filename}")
                print(f"   类别: {doc.category}")
                print(f"   关键词: {', '.join(doc.keywords[:5])}")
                print(f"   摘要: {doc.summary}")
                
                # 查找相似文档
                similar_docs = self.find_similar_documents(doc_id, limit=2)
                if similar_docs:
                    print("   相似文档:")
                    for sim in similar_docs:
                        print(f"     - {sim['filename']} (相似度: {sim['similarity']:.3f})")
        
        # 演示批量处理
        print("\n🎯 批量处理演示")
        print("=" * 50)
        
        batch_docs = self.batch_process_directory(sample_dir, category='未分类')
        print(f"✅ 批量处理完成: {len(batch_docs)} 个文档")
        
        # 演示分析报告
        print("\n🎯 分析报告演示")
        print("=" * 50)
        
        if doc_ids:
            report = self.generate_analysis_report(doc_ids[0])
            
            print("📊 分析报告:")
            print(f"   文档: {report['document_info']['filename']}")
            print(f"   类别: {report['document_info']['category']}")
            print(f"   词数: {report['statistics']['word_count']}")
            print(f"   关键词: {len(report['document_info']['keywords'])}")
            
            if report['similar_documents']:
                print("   相似文档分析:")
                for sim in report['similar_documents']:
                    print(f"     - {sim['filename']}: {sim['similarity']:.3f}")
        
        # 清理示例目录
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
        
        print("\n🎉 文档分析工具演示完成！")
        print("\n下一步：")
        print("   1. 运行分析服务: python analyzer_service.py")
        print("   2. 启动Web界面: python web_app.py")
        print("   3. 测试分析API: python test_analyzer.py")

def main():
    """主函数"""
    print("🚀 文档分析工具")
    print("=" * 60)
    
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        analyzer = DocumentAnalyzer()
        analyzer.demo_document_analyzer()
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
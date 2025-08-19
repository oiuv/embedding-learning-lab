#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2课：获取第一个文本向量
============================

本课程将教你如何实际获取文本的嵌入向量。

学习目标：
1. 配置API环境
2. 获取单个文本的向量
3. 理解向量格式和维度
4. 验证向量获取成功

"""

import os
import sys
from typing import List
import numpy as np
from openai import OpenAI

class FirstEmbedding:
    """第一个文本向量获取类"""
    
    def __init__(self, api_key: str = None):
        """初始化客户端"""
        try:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = "text-embedding-v4"
            self.dimensions = 1024
            print("✅ 客户端初始化成功！")
        except Exception as e:
            print(f"❌ 客户端初始化失败: {e}")
            print("请确保：")
            print("1. 已安装openai库: pip install openai")
            print("2. 已设置环境变量 DASHSCOPE_API_KEY")
            sys.exit(1)
    
    def get_single_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"❌ 获取嵌入失败: {e}")
            return []
    
    def analyze_embedding(self, text: str, embedding: List[float]):
        """分析嵌入向量的特征"""
        print(f"\n📊 文本: '{text}'")
        print(f"📏 向量维度: {len(embedding)}")
        print(f"🔢 向量类型: {type(embedding)}")
        print(f"📋 前5个值: {embedding[:5]}")
        print(f"📋 后5个值: {embedding[-5:]}")
        
        # 计算统计信息
        embedding_array = np.array(embedding)
        print(f"📈 统计信息:")
        print(f"   最小值: {np.min(embedding_array):.4f}")
        print(f"   最大值: {np.max(embedding_array):.4f}")
        print(f"   平均值: {np.mean(embedding_array):.4f}")
        print(f"   标准差: {np.std(embedding_array):.4f}")
    
    def compare_texts(self, texts: List[str]):
        """比较多个文本的嵌入向量"""
        print("\n🎯 比较多个文本的嵌入")
        print("=" * 50)
        
        embeddings = {}
        for text in texts:
            embedding = self.get_single_embedding(text)
            if embedding:
                embeddings[text] = embedding
                print(f"✅ 获取成功: '{text}' - 维度: {len(embedding)}")
        
        return embeddings
    
    def demonstrate_batch_processing(self, texts: List[str]):
        """演示批量处理"""
        print("\n🎯 批量获取嵌入向量")
        print("=" * 50)
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            embeddings = [data.embedding for data in response.data]
            print(f"✅ 批量获取成功！")
            print(f"📊 文本数量: {len(texts)}")
            print(f"📊 嵌入数量: {len(embeddings)}")
            print(f"📊 每个维度: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ 批量获取失败: {e}")
            return []
    
    def validate_embedding_quality(self, text: str, embedding: List[float]):
        """验证嵌入质量"""
        print("\n🎯 嵌入质量验证")
        print("=" * 50)
        
        # 检查基本属性
        checks = [
            ("维度正确", len(embedding) == self.dimensions),
            ("非空向量", len(embedding) > 0),
            ("数值类型", all(isinstance(x, (int, float)) for x in embedding)),
            ("合理范围", all(-2 <= x <= 2 for x in embedding))
        ]
        
        for check_name, result in checks:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{check_name}: {status}")
        
        # 检查向量范数
        norm = np.linalg.norm(embedding)
        print(f"📏 向量范数: {norm:.4f}")
        
        # 检查是否为有效向量
        if norm > 0.1:
            print("✅ 检测到有效向量")
        else:
            print("⚠️ 向量范数过小，可能存在问题")
    
    def save_embedding_example(self, text: str, embedding: List[float]):
        """保存嵌入示例"""
        print("\n🎯 保存嵌入示例")
        print("=" * 50)
        
        # 创建保存目录
        os.makedirs("01-basics/data", exist_ok=True)
        
        # 保存到文件
        filename = f"01-basics/data/first_embedding_{text.replace(' ', '_')[:20]}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"文本: {text}\n")
            f.write(f"维度: {len(embedding)}\n")
            f.write(f"向量: {embedding}\n")
        
        print(f"✅ 嵌入已保存到: {filename}")

def main():
    """主函数"""
    print("🚀 第2课：获取第一个文本向量")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量: export DASHSCOPE_API_KEY='你的密钥'")
        print("或使用: python 02-first-embedding.py --api-key 你的密钥")
        
        # 检查命令行参数
        if len(sys.argv) > 1 and sys.argv[1] == "--api-key":
            api_key = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            api_key = None
    else:
        api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 创建实例
    embedder = FirstEmbedding(api_key)
    
    # 获取第一个文本的嵌入
    first_text = "你好，世界！"
    print(f"🎯 获取第一个文本的嵌入向量: '{first_text}'")
    
    embedding = embedder.get_single_embedding(first_text)
    
    if embedding:
        # 分析嵌入
        embedder.analyze_embedding(first_text, embedding)
        embedder.validate_embedding_quality(first_text, embedding)
        embedder.save_embedding_example(first_text, embedding)
        
        # 比较多个文本
        sample_texts = [
            "机器学习",
            "深度学习",
            "人工智能",
            "自然语言处理"
        ]
        
        embeddings = embedder.compare_texts(sample_texts)
        
        # 批量处理
        batch_embeddings = embedder.demonstrate_batch_processing(sample_texts)
        
        print("\n🎉 第2课完成！")
        print("你已经学会了：")
        print("✅ 配置API环境")
        print("✅ 获取单个文本向量")
        print("✅ 批量获取文本向量")
        print("✅ 验证嵌入质量")
        print("✅ 保存嵌入结果")
        print("\n下一课：03-similarity-calculation.py - 计算文本相似度")
    else:
        print("❌ 获取嵌入失败，请检查配置")

if __name__ == "__main__":
    main()
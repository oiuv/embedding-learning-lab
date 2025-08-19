#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
官方示例整合 - text-embedding-v4最佳实践
=====================================

基于官方文档提供的三种核心使用方式：
1. 单文本向量化
2. 批量文本向量化  
3. 文件文本向量化

作者：官方文档 + 项目整合
"""

import os
from typing import List, Dict, Union
from openai import OpenAI
import json

class OfficialEmbeddingExamples:
    """官方示例整合类"""
    
    def __init__(self, api_key: str = None):
        """初始化客户端"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "text-embedding-v4"
        self.dimensions = 1024
    
    def example_single_text_embedding(self, text: str) -> Dict:
        """示例1：单文本向量化
        
        适用于：单个句子、段落、商品评价等场景
        """
        try:
            completion = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            result = {
                "text": text,
                "embedding": completion.data[0].embedding,
                "dimensions": len(completion.data[0].embedding),
                "model": completion.model,
                "usage": completion.usage.dict()
            }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def example_batch_text_embedding(self, texts: List[str]) -> Dict:
        """示例2：批量文本向量化
        
        适用于：文档集合、商品列表、评论批量处理等场景
        注意：单次最多10个文本
        """
        if len(texts) > 10:
            return {"error": "一次最多处理10个文本，请分批处理"}
            
        try:
            completion = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            results = []
            for i, data in enumerate(completion.data):
                results.append({
                    "text": texts[i],
                    "embedding": data.embedding,
                    "index": i
                })
            
            return {
                "results": results,
                "total": len(results),
                "model": completion.model,
                "usage": completion.usage.dict()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def example_file_text_embedding(self, file_path: str) -> Dict:
        """示例3：文件文本向量化
        
        适用于：长篇文档、多个段落、批量文本文件等场景
        注意：文件总行数不超过10行，每行不超过8192 Token
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取文件并按行分割
                lines = [line.strip() for line in f if line.strip()]
                
            if len(lines) > 10:
                return {"error": f"文件行数({len(lines)})超过10行限制"}
                
            completion = self.client.embeddings.create(
                model=self.model,
                input=lines,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            
            results = []
            for i, data in enumerate(completion.data):
                results.append({
                    "line": lines[i],
                    "embedding": data.embedding,
                    "line_number": i + 1
                })
            
            return {
                "file_path": file_path,
                "results": results,
                "total_lines": len(results),
                "model": completion.model,
                "usage": completion.usage.dict()
            }
            
        except FileNotFoundError:
            return {"error": f"文件 {file_path} 不存在"}
        except Exception as e:
            return {"error": str(e)}
    
    def example_different_dimensions(self, text: str) -> Dict:
        """示例4：不同维度的效果对比
        
        展示不同维度对同一文本的向量化效果
        """
        dimensions = [64, 128, 256, 512, 768, 1024, 1536, 2048]
        results = {}
        
        for dim in dimensions:
            try:
                completion = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    dimensions=dim,
                    encoding_format="float"
                )
                
                embedding = completion.data[0].embedding
                results[dim] = {
                    "embedding": embedding,
                    "norm": sum(x**2 for x in embedding) ** 0.5,
                    "memory_usage": len(embedding) * 4  # float32 = 4 bytes
                }
            except Exception as e:
                results[dim] = {"error": str(e)}
        
        return {
            "text": text,
            "dimension_comparison": results
        }
    
    def example_chinese_texts(self) -> Dict:
        """示例5：中文文本向量化示例
        
        展示中文文本的向量化效果，包括诗意表达、商品评价等
        """
        chinese_texts = [
            "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买",
            "风急天高猿啸哀，渚清沙白鸟飞回",
            "无边落木萧萧下，不尽长江滚滚来",
            "机器学习是人工智能的重要分支",
            "Python是最流行的数据科学语言",
            "这个商品性价比很高，值得推荐",
            "代码写得很好，逻辑清晰，注释完整"
        ]
        
        return self.example_batch_text_embedding(chinese_texts)
    
    def demo_all_examples(self):
        """演示所有官方示例"""
        print("🚀 官方示例演示 - text-embedding-v4")
        print("=" * 50)
        
        # 示例1：单文本
        print("\n📌 示例1：单文本向量化")
        single_result = self.example_single_text_embedding(
            "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢"
        )
        if "error" not in single_result:
            print(f"文本：{single_result['text']}")
            print(f"向量维度：{single_result['dimensions']}")
            print(f"向量范数：{sum(x**2 for x in single_result['embedding']) ** 0.5:.4f}")
        else:
            print(f"错误：{single_result['error']}")
        
        # 示例2：批量文本
        print("\n📌 示例2：批量文本向量化")
        batch_result = self.example_batch_text_embedding([
            "风急天高猿啸哀",
            "渚清沙白鸟飞回", 
            "无边落木萧萧下",
            "不尽长江滚滚来"
        ])
        if "error" not in batch_result:
            print(f"处理了{batch_result['total']}个文本")
            for item in batch_result['results']:
                print(f"  {item['text'][:10]}... -> 向量范数：{sum(x**2 for x in item['embedding']) ** 0.5:.4f}")
        else:
            print(f"错误：{batch_result['error']}")
        
        # 示例3：不同维度对比
        print("\n📌 示例3：不同维度效果对比")
        dim_result = self.example_different_dimensions("机器学习是人工智能的重要分支")
        if "dimension_comparison" in dim_result:
            print("维度 vs 内存占用：")
            for dim, data in dim_result["dimension_comparison"].items():
                if "error" not in data:
                    print(f"  {dim}维：{data['memory_usage']}字节")
        
        # 示例4：中文文本
        print("\n📌 示例4：中文文本向量化")
        chinese_result = self.example_chinese_texts()
        if "error" not in chinese_result:
            print(f"处理了{chinese_result['total']}个中文文本")

def create_sample_file():
    """创建示例文本文件"""
    sample_texts = [
        "人工智能正在改变我们的生活和工作方式",
        "机器学习是人工智能的重要分支，通过数据学习模式",
        "深度学习使用神经网络处理复杂问题，需要大量计算资源",
        "Python是最流行的数据科学语言，拥有丰富的库生态系统",
        "自然语言处理让计算机理解和处理人类语言"
    ]
    
    with open('data/sample_texts.txt', 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print("✅ 示例文件已创建：data/sample_texts.txt")

def main():
    """主函数"""
    # 创建必要的目录
    os.makedirs('data', exist_ok=True)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️ 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # 创建示例文件
    create_sample_file()
    
    # 运行演示
    examples = OfficialEmbeddingExamples()
    examples.demo_all_examples()
    
    print("\n🎉 官方示例演示完成！")
    print("\n使用方法：")
    print("1. 单文本：examples.example_single_text_embedding('你的文本')")
    print("2. 批量：examples.example_batch_text_embedding(['文本1', '文本2'])")
    print("3. 文件：examples.example_file_text_embedding('data/sample_texts.txt')")
    print("4. 维度对比：examples.example_different_dimensions('测试文本')")

if __name__ == "__main__":
    main()
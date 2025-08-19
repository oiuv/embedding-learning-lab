#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的嵌入客户端工具
====================

提供统一的文本嵌入获取接口，自动处理API限制。
"""

import os
from typing import List
from openai import OpenAI

class EmbeddingClient:
    """统一的嵌入客户端"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-v4", dimensions: int = 1024):
        """初始化客户端"""
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.dimensions = dimensions
        self.max_batch_size = 10  # API限制
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
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
        """批量获取文本嵌入向量，自动处理API限制"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # 分批处理以避免API限制
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i+self.max_batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                    encoding_format="float"
                )
                all_embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"❌ 批量获取失败: {e}")
                return []
        
        return all_embeddings
    
    def validate_batch_size(self, texts: List[str]) -> bool:
        """验证批处理大小是否合法"""
        return len(texts) <= self.max_batch_size
    
    def split_batch(self, texts: List[str]) -> List[List[str]]:
        """将文本列表分割成符合API限制的批次"""
        return [texts[i:i+self.max_batch_size] for i in range(0, len(texts), self.max_batch_size)]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试官方示例
================

运行官方提供的所有示例，验证API配置和功能
"""

import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.official_examples import OfficialEmbeddingExamples

def quick_test():
    """快速测试所有功能"""
    print("🧪 快速测试 - 官方示例验证")
    print("=" * 40)
    
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='你的密钥'")
        return False
    
    try:
        examples = OfficialEmbeddingExamples()
        
        # 测试单文本
        print("\n1️⃣ 测试单文本...")
        result = examples.example_single_text_embedding("你好，世界！")
        if "error" not in result:
            print(f"   ✅ 成功！维度：{result['dimensions']}")
        else:
            print(f"   ❌ 失败：{result['error']}")
            return False
        
        # 测试批量文本
        print("\n2️⃣ 测试批量文本...")
        batch_texts = ["你好", "世界", "人工智能", "机器学习"]
        result = examples.example_batch_text_embedding(batch_texts)
        if "error" not in result:
            print(f"   ✅ 成功！处理了{result['total']}个文本")
        else:
            print(f"   ❌ 失败：{result['error']}")
            return False
        
        # 测试文件处理
        print("\n3️⃣ 测试文件处理...")
        result = examples.example_file_text_embedding("data/sample_texts.txt")
        if "error" not in result:
            print(f"   ✅ 成功！处理了{result['total_lines']}行文本")
        else:
            print(f"   ❌ 失败：{result['error']}")
            return False
        
        print("\n🎉 所有测试通过！API配置正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

if __name__ == "__main__":
    quick_test()
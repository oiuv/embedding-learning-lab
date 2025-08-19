# 文本排序模型教程 - 高级应用

## 📚 课程概述

本教程专门介绍文本排序模型（Text ReRank Model）的高级应用，帮助您理解如何将排序技术集成到现有的语义搜索和问答系统中，显著提升搜索结果的相关性和准确性。

## 🎯 学习目标

完成本教程后，您将能够：

1. **理解文本排序原理**：掌握gte-rerank模型的工作机制和应用场景
2. **系统集成能力**：学会将排序模型无缝集成到现有系统中
3. **性能优化技巧**：掌握缓存、批处理等性能优化方法
4. **质量评估**：建立完善的排序质量评估体系
5. **成本控制**：合理规划API调用成本和资源使用

## 📁 文件结构

```
03-advanced/05-text-reranking/
├── README.md                          # 本说明文档
├── 01-text-rerank-basics.py          # 基础教程：排序模型入门
├── 02-integration-guide.py           # 集成指南：系统集成方法
├── 03-comparison-demo.py             # 对比演示：不同排序方法对比
├── 04-performance-benchmark.py       # 性能基准：全面性能测试
└── examples/                         # 实际应用示例
    ├── medical_search_demo.py        # 医疗搜索场景示例
    ├── legal_document_ranking.py     # 法律文书排序示例
    └── e-commerce_search.py          # 电商搜索优化示例
```

## 🚀 快速开始

### 环境要求

```bash
# 确保已设置API密钥
export DASHSCOPE_API_KEY="your-api-key"

# 安装依赖
pip install dashscope numpy pandas matplotlib seaborn
```

### 运行基础教程

```bash
# 运行基础排序教程
python 01-text-rerank-basics.py

# 运行系统集成演示
python 02-integration-guide.py

# 运行性能对比分析
python 03-comparison-demo.py

# 运行完整性能基准测试
python 04-performance-benchmark.py
```

## 📊 核心功能对比

| 功能特性 | 嵌入模型 | 文本排序模型 | 混合策略 |
|---------|----------|-------------|----------|
| **准确性** | 良好 | 优秀 | 最佳 |
| **处理速度** | 快速 | 中等 | 可配置 |
| **成本** | 低 | 中等 | 可控 |
| **适用场景** | 大规模筛选 | 精准排序 | 综合优化 |
| **缓存支持** | 支持 | 支持 | 完全支持 |

## 🎯 核心应用场景

### 1. 医疗文献搜索
- **场景描述**：在海量医疗文献中查找最相关的研究
- **优化效果**：相关性提升35-50%
- **推荐配置**：混合策略，缓存TTL=1小时

### 2. 法律文书检索
- **场景描述**：快速定位最相关的法律条文和案例
- **优化效果**：精确率提升40%
- **推荐配置**：文本排序模型，批处理优化

### 3. 电商商品推荐
- **场景描述**：根据用户查询排序商品结果
- **优化效果**：点击率提升25%
- **推荐配置**：混合策略，实时缓存

## ⚡ 性能优化建议

### 缓存策略

```python
# 推荐缓存配置
reranker = TextReranker(
    cache_enabled=True,
    cache_ttl=3600,  # 1小时
    max_documents=100
)
```

### 批处理优化

```python
# 批处理配置
batch_size = 50  # 中等规模推荐
large_scale = 100  # 大规模推荐
```

### 成本估算

| 文档数量 | 单次查询成本 | 推荐策略 |
|----------|-------------|----------|
| ≤10 | ¥0.0016 | 直接使用 |
| 11-50 | ¥0.008-0.04 | 混合策略 |
| 51-100 | ¥0.04-0.08 | 缓存优化 |
| ≥100 | ¥0.08+ | 分批处理 |

## 🔧 系统集成示例

### 1. 基础集成

```python
from utils.text_reranker import TextReranker

# 初始化排序器
reranker = TextReranker()

# 创建文档
documents = [
    RerankDocument(text="文档内容...", doc_id="doc1"),
    RerankDocument(text="文档内容...", doc_id="doc2")
]

# 执行排序
results = reranker.rerank("查询文本", documents)
```

### 2. 高级集成

```python
from utils.text_reranker import AdvancedReranker

# 高级排序器
advanced_reranker = AdvancedReranker()

# 混合排序（结合多种信号）
results = advanced_reranker.hybrid_rank(
    query="查询文本",
    documents=documents,
    query_embedding=query_embedding,
    doc_embeddings=doc_embeddings
)
```

### 3. 现有系统升级

```python
# 在现有搜索系统中添加排序优化
def enhanced_search(query, documents):
    # 1. 使用嵌入模型初步筛选
    candidates = embedding_based_filtering(query, documents, top_k=100)
    
    # 2. 使用文本排序模型精排
    reranked = reranker.rerank(query, candidates, top_n=10)
    
    return reranked
```

## 📈 性能基准

### 测试配置

| 规模 | 文档数 | 查询数 | 预期吞吐量 | 推荐缓存 |
|------|--------|--------|------------|----------|
| 小规模 | 10 | 5 | 20 docs/sec | 启用 |
| 中等规模 | 50 | 10 | 15 docs/sec | 启用 |
| 大规模 | 100 | 20 | 10 docs/sec | 必需 |
| 超大规模 | 200 | 30 | 5 docs/sec | 必需 |

### 缓存性能提升

- **首次调用**: 0.1-0.5秒
- **缓存命中**: 0.001-0.01秒
- **性能提升**: 10-50倍

## 🎓 学习路径

### 初学者路径
1. **第1步**：运行 `01-text-rerank-basics.py` 了解基础概念
2. **第2步**：阅读集成指南 `02-integration-guide.py`
3. **第3步**：尝试简单集成示例

### 进阶路径
1. **第1步**：完成对比演示 `03-comparison-demo.py`
2. **第2步**：运行性能基准测试 `04-performance-benchmark.py`
3. **第3步**：根据实际需求优化配置

### 专家路径
1. **第1步**：自定义排序策略
2. **第2步**：实现业务特定的评估指标
3. **第3步**：构建完整的监控体系

## 🔍 调试与故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥设置
   - 确认网络连接
   - 验证文档格式

2. **性能问题**
   - 启用缓存功能
   - 调整批处理大小
   - 检查并发限制

3. **成本过高**
   - 使用缓存减少重复调用
   - 优化文档预处理
   - 设置合理的top_n参数

### 调试工具

```python
# 快速性能检查
benchmark = PerformanceBenchmark()
quick_result = benchmark.quick_performance_check()
print(quick_result)

# 缓存状态检查
stats = reranker.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
```

## 🚀 下一步学习

完成本教程后，建议继续学习：

1. **实战项目**：在现有搜索系统中集成排序模型
2. **高级优化**：实现自适应排序策略
3. **监控体系**：建立完整的性能监控
4. **业务集成**：针对特定业务场景优化

## 📞 支持与反馈

- **问题反馈**：GitHub Issues
- **技术支持**：技术支持邮箱
- **社区讨论**：技术交流群

## 🔄 版本更新

- **v1.0**: 基础排序功能
- **v1.1**: 缓存优化
- **v1.2**: 批处理支持
- **v1.3**: 性能基准测试
- **v1.4**: 高级集成示例

---

🎉 **开始您的文本排序优化之旅吧！**
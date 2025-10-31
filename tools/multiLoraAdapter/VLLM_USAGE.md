# vLLM多LoRA推理使用说明

## 概述

基于vLLM库的多LoRA功能，我们提供了针对Qwen3-14B基础模型的不同秩LoRA矩阵融合和推理解决方案。

## 文件说明

### 核心脚本

1. **`loraMergeTest.py`** - 完整的跨秩LoRA融合和推理测试脚本
   - 支持SVD分解的跨秩LoRA融合
   - 集成vLLM多LoRA推理功能
   - 包含性能测试和验证

2. **`vllm_qwen_test.py`** - 简化的vLLM多LoRA推理测试脚本
   - 专门用于演示vLLM的多LoRA功能
   - 支持LoRA切换和并发使用
   - 更轻量级的实现

3. **`vllm_multilora_inference.py`** - 原始vLLM示例
   - vLLM官方提供的多LoRA推理示例
   - 基于Llama2模型的演示

## 安装依赖

### 1. 安装vLLM

```bash
# 安装vLLM（需要CUDA支持）
pip install vllm

# 或者从源码安装最新版本
pip install git+https://github.com/vllm-project/vllm.git
```

### 2. 安装其他依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装vLLM相关依赖
pip install -r requirements_vllm.txt
```

### 3. 系统要求

- **CUDA**: 11.8或更高版本
- **Python**: 3.8或更高版本
- **PyTorch**: 2.0或更高版本
- **GPU内存**: 至少16GB（推荐24GB+）

## 使用方法

### 1. 基础使用

```bash
# 运行完整的跨秩LoRA融合测试
python loraMergeTest.py

# 运行简化的vLLM多LoRA推理测试
python vllm_qwen_test.py
```

### 2. 编程方式使用

```python
from loraMergeTest import QwenLoRAMerger

# 创建融合器
merger = QwenLoRAMerger("./Qwen3-14B")

# 扫描LoRA适配器
lora_info = merger.scan_loras(".")

# 执行跨秩融合
merged_params = merger.merge_loras_cross_rank(
    lora_names=["lora1", "lora2", "lora3"],
    merge_weights=[0.4, 0.3, 0.3],
    target_rank=None,
    merge_strategy="svd"
)

# 创建vLLM引擎
engine = merger.create_vllm_engine(max_loras=4, max_lora_rank=64)

# 进行推理测试
test_prompts = merger.create_test_prompts(lora_paths)
merger.process_requests(engine, test_prompts)
```

## 功能特性

### 1. 跨秩LoRA融合

- **SVD分解**：使用奇异值分解将不同秩的LoRA矩阵调整到目标秩
- **权重归一化**：确保融合后的参数数值稳定
- **多种策略**：支持SVD、最大秩、最小秩三种融合策略

### 2. vLLM多LoRA推理

- **并发支持**：支持多个LoRA同时使用
- **动态切换**：支持运行时切换不同的LoRA适配器
- **内存优化**：高效的GPU内存管理

### 3. 性能测试

- **生成质量测试**：验证融合后的LoRA生成质量
- **性能基准测试**：测试不同配置下的性能表现
- **兼容性验证**：确保融合后的模型正常工作

## 配置参数

### vLLM引擎配置

```python
engine_args = EngineArgs(
    model="./Qwen3-14B",           # 基础模型路径
    enable_lora=True,              # 启用LoRA支持
    max_loras=4,                   # 最大同时使用的LoRA数量
    max_lora_rank=64,              # 最大LoRA秩
    max_cpu_loras=8,               # 最大CPU LoRA缓存数量
    max_num_seqs=256,              # 最大序列数
    trust_remote_code=True,        # 信任远程代码
    dtype="float16",               # 数据类型
    gpu_memory_utilization=0.8,    # GPU内存使用率
)
```

### LoRA融合配置

```python
# 融合策略
merge_strategy = "svd"        # "svd", "max_rank", "min_rank"

# 目标秩
target_rank = None            # None表示自动计算

# 融合权重
merge_weights = [0.4, 0.3, 0.3]  # 权重总和应为1.0
```

## 测试场景

### 1. 基础功能测试

- 扫描和加载LoRA适配器
- 跨秩LoRA融合
- 基础模型推理

### 2. LoRA功能测试

- 单个LoRA推理
- 多个LoRA切换
- 并发LoRA使用

### 3. 性能测试

- 融合质量验证
- 推理速度测试
- 内存使用监控

## 常见问题

### 1. 安装问题

**Q: vLLM安装失败**
A: 确保您的系统有CUDA支持，并且版本兼容。参考vLLM官方文档。

**Q: 内存不足**
A: 减少`max_loras`和`max_lora_rank`参数，或使用更小的模型。

### 2. 运行问题

**Q: LoRA加载失败**
A: 检查LoRA适配器路径和配置文件是否正确。

**Q: 推理速度慢**
A: 检查GPU内存使用情况，调整批处理大小。

### 3. 融合问题

**Q: 融合后模型质量下降**
A: 尝试不同的融合策略和权重设置。

**Q: 不同秩融合失败**
A: 确保使用SVD策略，并检查目标秩设置。

## 性能优化建议

### 1. 内存优化

- 使用`float16`数据类型
- 调整`gpu_memory_utilization`参数
- 合理设置`max_loras`和`max_lora_rank`

### 2. 速度优化

- 使用更大的批处理大小
- 启用Flash Attention（如果支持）
- 使用Tensor Parallelism（多GPU）

### 3. 质量优化

- 选择合适的融合策略
- 调整融合权重
- 验证融合后的模型质量

## 示例输出

```
基于vLLM的Qwen3-14B多LoRA推理测试
================================================================================

正在初始化vLLM Qwen测试器...
正在扫描LoRA适配器目录: .
发现LoRA适配器: qwen3-14b-lora-spatial-stats-final-v1 (r=16)
发现LoRA适配器: qwen3-14b-lora-spatial-stats-final-v2 (r=32)
发现LoRA适配器: qwen3-14b-lora-spatial-stats-final-v3 (r=64)
共发现 3 个LoRA适配器

发现 3 个LoRA适配器:
  qwen3-14b-lora-spatial-stats-final-v1: 秩=16, alpha=32
  qwen3-14b-lora-spatial-stats-final-v2: 秩=32, alpha=64
  qwen3-14b-lora-spatial-stats-final-v3: 秩=64, alpha=128

创建vLLM引擎...
正在创建vLLM引擎...
检测到最大LoRA秩: 64
vLLM引擎创建成功

开始处理vLLM多LoRA推理测试
================================================================================

请求 0 (基础模型):
输入: 请解释地理加权回归的基本原理：

输出: 地理加权回归（Geographically Weighted Regression, GWR）是一种空间统计分析方法...

请求 1 (LoRA: qwen3-14b-lora-spatial-stats-final-v1):
输入: 基于LoRA适配器qwen3-14b-lora-spatial-stats-final-v1，请解释地理加权回归的基本原理：

输出: 基于空间统计的LoRA适配器，地理加权回归是一种考虑空间异质性的回归分析方法...
```

## 总结

基于vLLM的多LoRA功能，我们成功实现了针对Qwen3-14B的跨秩LoRA融合和推理解决方案。这种方法不仅支持不同秩的LoRA矩阵融合，还提供了高效的推理性能，为多任务学习和模型集成提供了强大的工具。

# LoRA适配器

这是一个用于管理多个LoRA（Low-Rank Adaptation）适配器的Python脚本，专门用于地理加权回归、空间分析等领域的微调模型。

## 功能特性

- 🔄 **动态加载和切换**：支持在运行时加载和切换多个LoRA适配器
- 🎯 **专业领域优化**：针对地理加权回归和空间分析任务进行优化
- 💾 **内存管理**：支持LoRA适配器的加载和卸载，优化内存使用
- 🛡️ **错误处理**：完善的错误处理和日志记录机制
- 🔧 **易于使用**：简洁的API接口，支持批量操作
- 🔍 **智能搜索**：支持通过名称搜索和选择LoRA适配器
- 🖥️ **交互式界面**：提供用户友好的交互式选择界面

## 安装依赖

```bash
pip install -r requirements.txt
```

## 目录结构

```
multiLoraAdapter/
├── loraAdapter.py          # 主要的LoRA适配器类
├── loraMerge.py            # LoRA矩阵融合工具（同秩）
├── loraNormMerge.py        # 跨秩LoRA矩阵融合工具
├── example_usage.py        # 使用示例
├── example_merge.py        # 融合功能示例
├── example_norm_merge.py   # 跨秩融合功能示例
├── requirements.txt        # 依赖包列表
├── README.md              # 说明文档
├── Qwen3-14B/             # 基础模型目录
├── qwen3-14b-lora-spatial-stats-final-v1/  # LoRA适配器1
├── qwen3-14b-lora-spatial-stats-final-v2/  # LoRA适配器2
├── qwen3-14b-lora-spatial-stats-final-v3/  # LoRA适配器3
└── merged_models/          # 融合后的模型保存目录
```

## 快速开始

### 基本使用

```python
from loraAdapter import LoRAAdapter

# 初始化适配器
adapter = LoRAAdapter("./Qwen3-14B")

# 加载LoRA适配器
adapter.load_lora("./qwen3-14b-lora-spatial-stats-final-v1", "spatial_v1")
adapter.load_lora("./qwen3-14b-lora-spatial-stats-final-v2", "spatial_v2")

# 切换到指定LoRA适配器
adapter.switch_lora("spatial_v1")

# 生成文本
response = adapter.generate("请解释地理加权回归的基本原理：")
print(response)
```

### 批量加载LoRA适配器

```python
# 加载目录下所有LoRA适配器
results = adapter.load_all_loras("./")
for lora_name, success in results.items():
    print(f"加载 {lora_name}: {'成功' if success else '失败'}")
```

### 切换LoRA适配器

```python
# 获取已加载的LoRA适配器列表
loaded_loras = adapter.get_loaded_loras()
print(f"已加载的LoRA适配器: {loaded_loras}")

# 切换到不同的LoRA适配器
adapter.switch_lora("spatial_v2")

# 获取当前使用的LoRA适配器
current_lora = adapter.get_current_lora()
print(f"当前使用的LoRA: {current_lora}")
```

### 通过名称搜索和选择LoRA适配器

```python
# 通过精确名称切换
adapter.switch_lora_by_name("spatial_v1")

# 通过部分名称搜索（支持模糊匹配）
adapter.switch_lora_by_name("spatial")  # 会匹配包含"spatial"的LoRA适配器

# 搜索匹配的LoRA适配器
matches = adapter.find_lora_by_name("stats")
print(f"包含'stats'的LoRA适配器: {matches}")

# 交互式选择LoRA适配器
selected_lora = adapter.select_lora_interactive()
if selected_lora:
    adapter.switch_lora(selected_lora)
```

### LoRA矩阵融合

#### 同秩LoRA融合

```python
from loraMerge import LoRAMerger

# 初始化融合器
merger = LoRAMerger("./Qwen3-14B")

# 扫描LoRA适配器
lora_info = merger.scan_loras(".")

# 选择要融合的LoRA适配器
lora_names = ["qwen3-14b-lora-spatial-stats-final-v1", "qwen3-14b-lora-spatial-stats-final-v2"]

# 设置融合权重
weights = [0.6, 0.4]

# 执行融合
merged_model = merger.merge_loras(lora_names, weights)

# 保存融合后的模型
output_path = merger.save_merged_model(merged_model, "./merged_models", "Qwen3-14B-lora-v1")
```

#### 跨秩LoRA融合

```python
from loraNormMerge import LoRANormMerger

# 初始化跨秩融合器
merger = LoRANormMerger("./Qwen3-14B")

# 扫描LoRA适配器
lora_info = merger.scan_loras(".")

# 选择不同秩的LoRA适配器
lora_names = ["lora-rank16", "lora-rank32", "lora-rank64"]

# 设置融合权重
weights = [0.5, 0.3, 0.2]

# 使用SVD策略进行跨秩融合
merged_model = merger.merge_loras_cross_rank(
    lora_names, 
    weights, 
    target_rank=None,  # 自动计算
    merge_strategy="svd"
)

# 保存融合后的模型
output_path = merger.save_merged_model(merged_model, "./merged_models", "Qwen3-14B-lora-crossrank")
```

### 交互式使用

运行主程序进入交互式界面：

```bash
# LoRA适配器管理
python loraAdapter.py

# 同秩LoRA矩阵融合
python loraMerge.py

# 跨秩LoRA矩阵融合
python loraNormMerge.py
```

LoRA适配器管理界面提供以下功能：
1. 选择LoRA适配器（支持数字选择和名称搜索）
2. 通过名称搜索LoRA适配器
3. 生成文本
4. 显示LoRA适配器信息
5. 退出程序

同秩LoRA矩阵融合界面提供以下功能：
1. 扫描和显示可用的LoRA适配器
2. 按秩分组选择LoRA适配器
3. 设置融合权重
4. 选择融合方法（标准/安全）
5. 执行融合并保存模型
6. 测试融合后的模型

跨秩LoRA矩阵融合界面提供以下功能：
1. 扫描和显示可用的LoRA适配器（支持不同秩）
2. 跨秩选择LoRA适配器
3. 设置融合权重
4. 选择融合策略（SVD/最大秩/最小秩）
5. 设置目标融合秩
6. 执行跨秩融合并保存模型
7. 测试融合后的模型

## API参考

### LoRAAdapter类

#### 初始化
```python
LoRAAdapter(base_model_path: str, device: str = "auto")
```

#### 主要方法

- `load_lora(lora_path: str, lora_name: str = None) -> bool`
  - 加载LoRA适配器
  
- `switch_lora(lora_name: str) -> bool`
  - 切换到指定的LoRA适配器
  
- `unload_lora(lora_name: str) -> bool`
  - 卸载指定的LoRA适配器
  
- `generate(prompt: str, max_length: int = 512, **kwargs) -> str`
  - 生成文本
  
- `get_loaded_loras() -> List[str]`
  - 获取已加载的LoRA适配器列表
  
- `get_current_lora() -> Optional[str]`
  - 获取当前使用的LoRA适配器名称
  
- `get_lora_info(lora_name: str) -> Optional[Dict]`
  - 获取LoRA适配器配置信息

- `find_lora_by_name(search_name: str) -> List[str]`
  - 根据名称搜索LoRA适配器（支持部分匹配）

- `switch_lora_by_name(lora_name: str) -> bool`
  - 通过名称切换LoRA适配器（支持部分匹配）

- `select_lora_interactive() -> Optional[str]`
  - 交互式选择LoRA适配器

### LoRAMerger类

#### 初始化
```python
LoRAMerger(base_model_path: str, device: str = "auto")
```

#### 主要方法

- `scan_loras(lora_dir: str) -> Dict[str, Dict]`
  - 扫描指定目录下的所有LoRA适配器

- `find_compatible_loras(target_r: int = None) -> List[str]`
  - 查找兼容的LoRA适配器（同秩）

- `merge_loras(lora_names: List[str], merge_weights: List[float] = None) -> PeftModel`
  - 融合多个LoRA适配器

- `save_merged_model(merged_model: PeftModel, output_path: str, model_name: str) -> str`
  - 保存融合后的模型

- `interactive_select_loras() -> List[str]`
  - 交互式选择LoRA适配器

- `interactive_set_weights(lora_names: List[str]) -> List[float]`
  - 交互式设置融合权重

### LoRANormMerger类

#### 初始化
```python
LoRANormMerger(base_model_path: str, device: str = "auto")
```

#### 主要方法

- `scan_loras(lora_dir: str) -> Dict[str, Dict]`
  - 扫描指定目录下的所有LoRA适配器

- `merge_loras_cross_rank(lora_names: List[str], merge_weights: List[float] = None, 
                         target_rank: int = None, merge_strategy: str = "svd") -> PeftModel`
  - 跨秩融合多个LoRA适配器

- `_svd_decompose(matrix: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, torch.Tensor]`
  - 使用SVD分解矩阵到目标秩

- `_normalize_lora_weights(lora_params: Dict[str, torch.Tensor], 
                          target_rank: int, alpha: float) -> Dict[str, torch.Tensor]`
  - 归一化LoRA权重到目标秩

- `_compute_effective_rank(lora_params: Dict[str, torch.Tensor]) -> int`
  - 计算LoRA参数的有效秩

- `_weighted_merge_loras(lora_params_list: List[Dict[str, torch.Tensor]], 
                        merge_weights: List[float], target_rank: int) -> Dict[str, torch.Tensor]`
  - 加权融合多个LoRA参数

- `save_merged_model(merged_model: PeftModel, output_path: str, model_name: str) -> str`
  - 保存融合后的模型

- `interactive_select_loras() -> List[str]`
  - 交互式选择LoRA适配器（支持跨秩）

- `interactive_set_weights(lora_names: List[str]) -> List[float]`
  - 交互式设置融合权重

## 使用示例

运行完整的使用示例：

```bash
# LoRA适配器管理示例
python example_usage.py

# 同秩LoRA矩阵融合示例
python example_merge.py

# 跨秩LoRA矩阵融合示例
python example_norm_merge.py
```

示例包括：
- 基本使用演示
- LoRA适配器切换
- 同秩LoRA矩阵融合
- 跨秩LoRA矩阵融合
- 专业任务处理（地理加权回归、空间分析）
- 错误处理演示

## 专业领域应用

### 地理加权回归（GWR）
- 模型原理解释
- 带宽参数选择
- R语言实现指导

### 空间分析
- 空间自相关分析
- 空间权重矩阵构建
- 空间聚类方法

## 注意事项

1. **内存管理**：LoRA适配器会占用内存，建议及时卸载不需要的适配器
2. **设备兼容性**：支持CPU、CUDA和MPS设备
3. **模型兼容性**：当前版本基于Qwen3-14B模型，其他模型可能需要调整配置
4. **LoRA配置**：确保LoRA适配器的配置与基础模型兼容

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查基础模型路径是否正确
   - 确认有足够的内存空间

2. **LoRA适配器加载失败**
   - 检查LoRA适配器路径和配置文件
   - 确认LoRA适配器与基础模型兼容

3. **设备相关错误**
   - 检查CUDA是否可用
   - 尝试使用CPU模式

### 日志调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 许可证

本项目遵循MIT许可证。


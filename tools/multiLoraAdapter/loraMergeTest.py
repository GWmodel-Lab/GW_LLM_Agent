#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于vLLM的多LoRA推理测试脚本
针对Qwen3-14B基础模型，实现不同秩LoRA矩阵的融合和推理测试

基于vLLM库的多LoRA功能，支持不同秩的LoRA适配器进行参数融合和推理
"""

import os
import json
import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import torch
import numpy as np

# 兼容性处理：修复 typing_extensions Sentinel 问题
import sys

# 在导入其他包之前修复 typing_extensions
if 'typing_extensions' in sys.modules:
    typing_extensions = sys.modules['typing_extensions']
    if not hasattr(typing_extensions, 'Sentinel'):
        class Sentinel:
            def __init__(self, name):
                self.name = name
            def __repr__(self):
                return f"<Sentinel: {self.name}>"
        typing_extensions.Sentinel = Sentinel

try:
    from typing_extensions import Sentinel
except ImportError:
    # 如果仍然无法导入，使用替代方案
    class Sentinel:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"<Sentinel: {self.name}>"

# vLLM 0.11.0 兼容性导入
try:
    from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    # 尝试从子模块导入
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        # 最后的回退方案
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams
        from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenLoRAMerger:
    """
    基于vLLM的Qwen3-14B LoRA融合器
    支持不同秩的LoRA适配器进行参数融合和推理
    """
    
    def __init__(self, base_model_path: str, device: str = "auto"):
        """
        初始化Qwen LoRA融合器
        
        Args:
            base_model_path: Qwen3-14B基础模型路径
            device: 设备类型
        """
        self.base_model_path = Path(base_model_path)
        self.device = self._get_device(device)
        self.base_model = None
        self.tokenizer = None
        self.available_loras = {}
        self.merged_loras = {}
        
        # 初始化基础模型
        self._load_base_model()
        
    def _get_device(self, device: str) -> str:
        """获取可用的设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_base_model(self):
        """加载Qwen3-14B基础模型和分词器"""
        try:
            logger.info(f"正在加载Qwen3-14B基础模型: {self.base_model_path}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
            
            # 加载基础模型
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            logger.info("Qwen3-14B基础模型加载成功")
            
        except Exception as e:
            logger.error(f"加载Qwen3-14B基础模型失败: {e}")
            raise
    
    def scan_loras(self, lora_dir: str) -> Dict[str, Dict]:
        """
        扫描指定目录下的所有LoRA适配器
        
        Args:
            lora_dir: LoRA适配器目录
            
        Returns:
            Dict[str, Dict]: LoRA适配器信息字典
        """
        lora_dir = Path(lora_dir)
        lora_info = {}
        
        if not lora_dir.exists():
            logger.error(f"LoRA目录不存在: {lora_dir}")
            return lora_info
        
        logger.info(f"正在扫描LoRA适配器目录: {lora_dir}")
        
        # 查找所有LoRA适配器目录
        for item in lora_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                try:
                    # 读取LoRA配置
                    config_path = item / "adapter_config.json"
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # 检查LoRA配置
                    if 'r' in config and 'target_modules' in config:
                        lora_info[item.name] = {
                            'path': str(item),
                            'config': config,
                            'r': config.get('r'),
                            'lora_alpha': config.get('lora_alpha'),
                            'target_modules': config.get('target_modules'),
                            'lora_dropout': config.get('lora_dropout'),
                            'bias': config.get('bias')
                        }
                        logger.info(f"发现LoRA适配器: {item.name} (r={config.get('r')})")
                    else:
                        logger.warning(f"LoRA适配器 {item.name} 配置不完整，跳过")
                        
                except Exception as e:
                    logger.error(f"读取LoRA适配器 {item.name} 配置失败: {e}")
                    continue
        
        self.available_loras = lora_info
        logger.info(f"共发现 {len(lora_info)} 个LoRA适配器")
        return lora_info
    
    def _svd_decompose(self, matrix: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用SVD分解矩阵到目标秩
        
        Args:
            matrix: 输入矩阵
            target_rank: 目标秩
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 分解后的矩阵A和B
        """
        try:
            # 执行SVD分解
            U, S, V = torch.svd(matrix)
            
            # 截断到目标秩
            U_truncated = U[:, :target_rank]
            S_truncated = S[:target_rank]
            V_truncated = V[:, :target_rank]
            
            # 重新组合为A和B的形式
            A = U_truncated @ torch.diag(torch.sqrt(S_truncated))
            B = torch.diag(torch.sqrt(S_truncated)) @ V_truncated.T
            
            return A, B
            
        except Exception as e:
            logger.error(f"SVD分解失败: {e}")
            raise
    
    def _normalize_lora_weights(self, lora_params: Dict[str, torch.Tensor], 
                               target_rank: int, alpha: float) -> Dict[str, torch.Tensor]:
        """
        归一化LoRA权重到目标秩
        
        Args:
            lora_params: LoRA参数字典
            target_rank: 目标秩
            alpha: LoRA alpha参数
            
        Returns:
            Dict[str, torch.Tensor]: 归一化后的参数字典
        """
        normalized_params = {}
        
        for param_name, param_tensor in lora_params.items():
            if 'lora_A' in param_name:
                # 处理lora_A参数
                if param_tensor.shape[1] != target_rank:
                    # 需要调整秩
                    A, B = self._svd_decompose(param_tensor, target_rank)
                    normalized_params[param_name] = A
                    # 找到对应的lora_B参数
                    b_name = param_name.replace('lora_A', 'lora_B')
                    if b_name in lora_params:
                        normalized_params[b_name] = B
                else:
                    normalized_params[param_name] = param_tensor
                    
            elif 'lora_B' in param_name:
                # 处理lora_B参数
                if param_name not in normalized_params:  # 如果还没有被A处理过
                    if param_tensor.shape[0] != target_rank:
                        # 需要调整秩
                        A, B = self._svd_decompose(param_tensor.T, target_rank)
                        normalized_params[param_name] = B
                        # 找到对应的lora_A参数
                        a_name = param_name.replace('lora_B', 'lora_A')
                        if a_name in lora_params:
                            normalized_params[a_name] = A
                    else:
                        normalized_params[param_name] = param_tensor
            else:
                # 其他参数直接复制
                normalized_params[param_name] = param_tensor
        
        return normalized_params
    
    def merge_loras_cross_rank(self, lora_names: List[str], merge_weights: List[float] = None, 
                              target_rank: int = None, merge_strategy: str = "svd") -> Dict[str, torch.Tensor]:
        """
        跨秩融合多个LoRA适配器，返回融合后的参数字典
        
        Args:
            lora_names: LoRA适配器名称列表
            merge_weights: 融合权重列表
            target_rank: 目标秩
            merge_strategy: 融合策略
            
        Returns:
            Dict[str, torch.Tensor]: 融合后的参数字典
        """
        if not lora_names:
            raise ValueError("至少需要指定一个LoRA适配器")
        
        if merge_weights is None:
            merge_weights = [1.0 / len(lora_names)] * len(lora_names)
        
        logger.info(f"开始跨秩融合LoRA适配器: {lora_names}")
        logger.info(f"融合权重: {merge_weights}")
        logger.info(f"融合策略: {merge_strategy}")
        
        # 检查所有LoRA适配器是否存在
        for name in lora_names:
            if name not in self.available_loras:
                raise ValueError(f"LoRA适配器 {name} 不存在")
        
        # 加载所有LoRA适配器
        lora_params_list = []
        ranks = []
        
        for lora_name in lora_names:
            lora_path = self.available_loras[lora_name]['path']
            
            # 加载LoRA模型
            lora_model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # 提取LoRA参数
            lora_params = {}
            for name, param in lora_model.named_parameters():
                if 'lora' in name.lower():
                    lora_params[name] = param.data.clone()
            
            lora_params_list.append(lora_params)
            ranks.append(self.available_loras[lora_name]['r'])
        
        # 确定目标秩
        if target_rank is None:
            if merge_strategy == "max_rank":
                target_rank = max(ranks)
            elif merge_strategy == "min_rank":
                target_rank = min(ranks)
            elif merge_strategy == "svd":
                # 使用SVD策略，选择最常见的秩
                from collections import Counter
                rank_counts = Counter(ranks)
                target_rank = rank_counts.most_common(1)[0][0]
            else:
                target_rank = max(ranks)
        
        logger.info(f"目标融合秩: {target_rank}")
        
        # 归一化所有LoRA参数到目标秩
        normalized_params_list = []
        for i, (lora_params, lora_name) in enumerate(zip(lora_params_list, lora_names)):
            alpha = self.available_loras[lora_name]['lora_alpha']
            normalized_params = self._normalize_lora_weights(lora_params, target_rank, alpha)
            normalized_params_list.append(normalized_params)
        
        # 融合归一化后的参数
        merged_params = self._weighted_merge_loras(normalized_params_list, merge_weights, target_rank)
        
        # 存储融合后的LoRA信息
        self.merged_loras[f"merged_{target_rank}"] = {
            'params': merged_params,
            'target_rank': target_rank,
            'source_loras': lora_names,
            'merge_weights': merge_weights
        }
        
        logger.info("跨秩LoRA融合完成")
        return merged_params
    
    def _weighted_merge_loras(self, lora_params_list: List[Dict[str, torch.Tensor]], 
                             merge_weights: List[float], target_rank: int) -> Dict[str, torch.Tensor]:
        """
        加权融合多个LoRA参数
        
        Args:
            lora_params_list: LoRA参数列表
            merge_weights: 融合权重列表
            target_rank: 目标秩
            
        Returns:
            Dict[str, torch.Tensor]: 融合后的参数字典
        """
        # 获取所有参数名称
        all_param_names = set()
        for lora_params in lora_params_list:
            all_param_names.update(lora_params.keys())
        
        merged_params = {}
        
        for param_name in all_param_names:
            # 收集所有LoRA中该参数的值
            param_values = []
            valid_weights = []
            
            for lora_params, weight in zip(lora_params_list, merge_weights):
                if param_name in lora_params:
                    param_values.append(lora_params[param_name])
                    valid_weights.append(weight)
            
            if not param_values:
                continue
            
            # 检查所有参数是否具有相同的形状
            shapes = [p.shape for p in param_values]
            if len(set(shapes)) == 1:
                # 形状相同，直接加权平均
                merged_param = torch.zeros_like(param_values[0])
                for param_val, weight in zip(param_values, valid_weights):
                    merged_param += param_val * weight
                merged_params[param_name] = merged_param
            else:
                # 形状不同，选择权重最大的参数
                logger.warning(f"参数 {param_name} 形状不一致，选择权重最大的参数")
                max_weight_idx = valid_weights.index(max(valid_weights))
                merged_params[param_name] = param_values[max_weight_idx]
        
        return merged_params
    
    def create_vllm_engine(self, max_loras: int = 4, max_lora_rank: int = 64, 
                          max_cpu_loras: int = 8) -> LLMEngine:
        """
        创建vLLM引擎，支持多LoRA推理
        
        Args:
            max_loras: 最大同时使用的LoRA数量
            max_lora_rank: 最大LoRA秩
            max_cpu_loras: 最大CPU LoRA缓存数量
            
        Returns:
            LLMEngine: vLLM引擎实例
        """
        try:
            logger.info("正在创建vLLM引擎...")
            
            engine_args = EngineArgs(
                model=str(self.base_model_path),
                enable_lora=True,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                max_cpu_loras=max_cpu_loras,
                max_num_seqs=256,
                trust_remote_code=True,
                dtype="float16" if self.device == "cuda" else "float32",
                gpu_memory_utilization=0.8,
                tensor_parallel_size=1,
                pipeline_parallel_size=1
            )
            
            engine = LLMEngine.from_engine_args(engine_args)
            logger.info("vLLM引擎创建成功")
            return engine
            
        except Exception as e:
            logger.error(f"创建vLLM引擎失败: {e}")
            raise
    
    def create_test_prompts(self, lora_paths: Dict[str, str]) -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
        """
        创建测试提示词，包含基础模型和不同LoRA的请求
        
        Args:
            lora_paths: LoRA路径字典
            
        Returns:
            List[Tuple[str, SamplingParams, Optional[LoRARequest]]]: 测试提示词列表
        """
        prompts = []
        
        # 基础模型请求
        prompts.extend([
            (
                "请解释地理加权回归的基本原理：",
                SamplingParams(
                    temperature=0.7, 
                    top_p=0.9, 
                    max_tokens=200
                ),
                None,
            ),
            (
                "什么是空间自相关分析？",
                SamplingParams(
                    temperature=0.8, 
                    top_k=5, 
                    presence_penalty=0.2, 
                    max_tokens=150
                ),
                None,
            ),
        ])
        
        # LoRA请求
        for i, (lora_name, lora_path) in enumerate(lora_paths.items(), 1):
            prompts.extend([
                (
                    f"基于LoRA适配器{lora_name}，请解释地理加权回归的基本原理：",
                    SamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=200,
                    ),
                    LoRARequest(lora_name, i, lora_path),
                ),
                (
                    f"使用{lora_name}适配器，什么是空间自相关分析？",
                    SamplingParams(
                        temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        max_tokens=150,
                    ),
                    LoRARequest(lora_name, i, lora_path),
                ),
            ])
        
        return prompts
    
    def process_requests(self, engine: LLMEngine, 
                        test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest]]]):
        """
        处理请求并显示结果
        
        Args:
            engine: vLLM引擎
            test_prompts: 测试提示词列表
        """
        request_id = 0
        
        print("=" * 80)
        print("开始处理LoRA融合测试请求")
        print("=" * 80)
        
        while test_prompts or engine.has_unfinished_requests():
            if test_prompts:
                prompt, sampling_params, lora_request = test_prompts.pop(0)
                
                # 显示请求信息
                lora_info = f" (LoRA: {lora_request.lora_name})" if lora_request else " (基础模型)"
                print(f"\n请求 {request_id}{lora_info}:")
                print(f"输入: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                
                engine.add_request(
                    str(request_id), prompt, sampling_params, lora_request=lora_request
                )
                request_id += 1
            
            request_outputs: List[RequestOutput] = engine.step()
            
            for request_output in request_outputs:
                if request_output.finished:
                    print(f"\n输出: {request_output.outputs[0].text}")
                    print("-" * 80)
    
    def test_merged_lora_performance(self, merged_params: Dict[str, torch.Tensor], 
                                   test_prompts: List[str]) -> Dict[str, str]:
        """
        测试融合后的LoRA性能
        
        Args:
            merged_params: 融合后的参数字典
            test_prompts: 测试提示词列表
            
        Returns:
            Dict[str, str]: 测试结果
        """
        logger.info("开始测试融合后的LoRA性能...")
        
        # 创建融合后的LoRA配置
        target_rank = merged_params[list(merged_params.keys())[0]].shape[1] if 'lora_A' in list(merged_params.keys())[0] else 16
        
        merged_config = LoraConfig(
            r=target_rank,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 创建融合后的模型
        merged_model = get_peft_model(self.base_model, merged_config)
        
        # 加载融合后的参数
        for name, param in merged_model.named_parameters():
            if name in merged_params:
                param.data.copy_(merged_params[name])
        
        # 测试生成
        results = {}
        for i, prompt in enumerate(test_prompts):
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = merged_model.generate(
                        inputs,
                        max_length=200,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt):].strip()
                results[f"prompt_{i}"] = generated_text
                
            except Exception as e:
                logger.error(f"测试提示词 {i} 失败: {e}")
                results[f"prompt_{i}"] = f"生成失败: {e}"
        
        return results


def main():
    """主函数，演示基于vLLM的跨秩LoRA融合和推理"""
    # 设置路径
    base_model_path = "./Qwen3-14B"
    lora_dir = "."
    
    print("基于vLLM的Qwen3-14B跨秩LoRA融合测试")
    print("=" * 80)
    
    try:
        # 创建LoRA融合器
        print("正在初始化Qwen LoRA融合器...")
        merger = QwenLoRAMerger(base_model_path)
        
        # 扫描LoRA适配器
        print("正在扫描LoRA适配器...")
        lora_info = merger.scan_loras(lora_dir)
        
        if not lora_info:
            print("没有找到可用的LoRA适配器")
            return
        
        # 显示发现的LoRA适配器
        print(f"\n发现 {len(lora_info)} 个LoRA适配器:")
        for name, info in lora_info.items():
            print(f"  {name}: 秩={info['r']}, alpha={info['lora_alpha']}")
        
        # 选择要融合的LoRA适配器
        lora_names = list(lora_info.keys())[:3]  # 选择前3个
        print(f"\n选择融合的LoRA适配器: {lora_names}")
        
        # 设置融合权重
        weights = [0.4, 0.3, 0.3]
        print(f"融合权重: {weights}")
        
        # 执行跨秩融合
        print("\n开始跨秩融合...")
        merged_params = merger.merge_loras_cross_rank(
            lora_names, 
            weights, 
            target_rank=None,
            merge_strategy="svd"
        )
        
        print("跨秩融合完成！")
        
        # 测试融合后的LoRA性能
        test_prompts = [
            "请解释地理加权回归的基本原理：",
            "什么是空间自相关分析？",
            "如何选择合适的带宽参数？"
        ]
        
        print("\n测试融合后的LoRA性能...")
        performance_results = merger.test_merged_lora_performance(merged_params, test_prompts)
        
        print("\n性能测试结果:")
        for prompt_id, result in performance_results.items():
            print(f"\n{prompt_id}: {result}")
        
        # 创建vLLM引擎进行推理测试
        print("\n创建vLLM引擎进行推理测试...")
        engine = merger.create_vllm_engine(max_loras=4, max_lora_rank=64)
        
        # 准备LoRA路径
        lora_paths = {name: info['path'] for name, info in lora_info.items()}
        
        # 创建测试提示词
        test_prompts = merger.create_test_prompts(lora_paths)
        
        # 处理请求
        merger.process_requests(engine, test_prompts)
        
        print("\n所有测试完成！")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

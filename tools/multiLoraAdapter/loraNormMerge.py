#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于vLLM的通用LoRA矩阵融合脚本
支持不同秩的LoRA矩阵进行参数融合，使用SVD和权重归一化技术
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRANormMerger:
    """
    基于vLLM方法的通用LoRA矩阵融合类
    支持不同秩的LoRA适配器进行参数融合
    """
    
    def __init__(self, base_model_path: str, device: str = "auto"):
        """
        初始化LoRA归一化融合器
        
        Args:
            base_model_path: 基础模型路径
            device: 设备类型 ("auto", "cpu", "cuda", "mps")
        """
        self.base_model_path = Path(base_model_path)
        self.device = self._get_device(device)
        self.base_model = None
        self.tokenizer = None
        self.available_loras = {}  # 存储可用的LoRA适配器信息
        
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
        """加载基础模型和分词器"""
        try:
            logger.info(f"正在加载基础模型: {self.base_model_path}")
            
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
            
            logger.info("基础模型加载成功")
            
        except Exception as e:
            logger.error(f"加载基础模型失败: {e}")
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
    
    def load_lora(self, lora_path: str) -> PeftModel:
        """
        加载单个LoRA适配器
        
        Args:
            lora_path: LoRA适配器路径
            
        Returns:
            PeftModel: 加载的LoRA模型
        """
        try:
            lora_model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            return lora_model
        except Exception as e:
            logger.error(f"加载LoRA适配器失败 {lora_path}: {e}")
            raise
    
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
            # 验证输入
            if len(matrix.shape) != 2:
                raise ValueError(f"矩阵必须是2D的，当前形状: {matrix.shape}")
            
            if target_rank <= 0:
                raise ValueError(f"目标秩必须大于0，当前值: {target_rank}")
            
            # 确保目标秩不超过矩阵的最小维度
            min_dim = min(matrix.shape)
            actual_target_rank = min(target_rank, min_dim)
            
            if actual_target_rank != target_rank:
                logger.warning(f"目标秩 {target_rank} 超过矩阵最小维度 {min_dim}，调整为 {actual_target_rank}")
                target_rank = actual_target_rank
            
            # 执行SVD分解
            U, S, V = torch.svd(matrix)
            
            # 截断到目标秩
            U_truncated = U[:, :target_rank]
            S_truncated = S[:target_rank]
            V_truncated = V[:, :target_rank]
            
            # 重新组合为A和B的形式
            # 使用sqrt(S)来平衡A和B的权重
            sqrt_S = torch.sqrt(torch.clamp(S_truncated, min=1e-8))  # 避免数值问题
            A = U_truncated @ torch.diag(sqrt_S)
            B = torch.diag(sqrt_S) @ V_truncated.T
            
            return A, B
            
        except Exception as e:
            logger.error(f"SVD分解失败: {e}")
            logger.error(f"矩阵形状: {matrix.shape}, 目标秩: {target_rank}")
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
        
        # 按参数对处理LoRA参数
        processed_pairs = set()
        
        for param_name, param_tensor in lora_params.items():
            if param_name in processed_pairs:
                continue
                
            if 'lora_A' in param_name:
                # 找到对应的lora_B参数
                b_name = param_name.replace('lora_A', 'lora_B')
                if b_name in lora_params:
                    A_orig = param_tensor
                    B_orig = lora_params[b_name]
                    
                    # 确定LoRA参数的形状模式
                    # 检查A和B的维度关系
                    if A_orig.shape[1] == B_orig.shape[0]:
                        # 模式1: A[hidden, rank], B[rank, hidden]
                        A_shape = A_orig.shape
                        B_shape = B_orig.shape
                        current_rank = A_orig.shape[1]
                    elif A_orig.shape[0] == B_orig.shape[1]:
                        # 模式2: A[rank, hidden], B[hidden, rank]
                        A_shape = A_orig.shape
                        B_shape = B_orig.shape
                        current_rank = A_orig.shape[0]
                    else:
                        # 维度不匹配，跳过
                        logger.warning(f"LoRA参数对维度不匹配: {param_name} {A_orig.shape} vs {b_name} {B_orig.shape}")
                        normalized_params[param_name] = param_tensor
                        if b_name in lora_params:
                            normalized_params[b_name] = lora_params[b_name]
                        continue
                    
                    # 如果秩已经匹配，直接复制
                    if current_rank == target_rank:
                        normalized_params[param_name] = A_orig
                        normalized_params[b_name] = B_orig
                    else:
                        # 需要调整秩
                        try:
                            # 重建完整的LoRA矩阵
                            if A_orig.shape[1] == B_orig.shape[0]:
                                # 模式1: A[hidden, rank] @ B[rank, hidden]
                                full_matrix = A_orig @ B_orig
                                logger.debug(f"模式1重建: A{A_orig.shape} @ B{B_orig.shape} = {full_matrix.shape}")
                            elif A_orig.shape[0] == B_orig.shape[1]:
                                # 模式2: A[rank, hidden] @ B[hidden, rank]
                                full_matrix = A_orig @ B_orig
                                logger.debug(f"模式2重建: A{A_orig.shape} @ B{B_orig.shape} = {full_matrix.shape}")
                            else:
                                # 维度完全不匹配，跳过SVD分解
                                logger.warning(f"LoRA参数对维度完全不匹配: A{A_orig.shape} vs B{B_orig.shape}")
                                normalized_params[param_name] = A_orig
                                normalized_params[b_name] = B_orig
                                continue
                            
                            # 对完整矩阵进行SVD分解
                            A_new, B_new = self._svd_decompose(full_matrix, target_rank)
                            
                            # 根据原始模式调整输出
                            if A_orig.shape[1] == B_orig.shape[0]:
                                # 模式1: 输出A[hidden, rank], B[rank, hidden]
                                normalized_params[param_name] = A_new
                                normalized_params[b_name] = B_new
                                logger.debug(f"模式1输出: A{A_new.shape}, B{B_new.shape}")
                            else:
                                # 模式2: 输出A[rank, hidden], B[hidden, rank]
                                normalized_params[param_name] = A_new
                                normalized_params[b_name] = B_new
                                logger.debug(f"模式2输出: A{A_new.shape}, B{B_new.shape}")
                                
                        except Exception as e:
                            logger.warning(f"SVD分解失败 {param_name}: {e}，使用简单调整")
                            # 使用简单的截断或填充
                            if A_orig.shape[1] == B_orig.shape[0]:
                                # 模式1: A[hidden, rank], B[rank, hidden]
                                if current_rank > target_rank:
                                    normalized_params[param_name] = A_orig[:, :target_rank]
                                    normalized_params[b_name] = B_orig[:target_rank, :]
                                else:
                                    # 填充
                                    A_padding = torch.zeros(A_orig.shape[0], target_rank - current_rank, 
                                                          dtype=A_orig.dtype, device=A_orig.device)
                                    B_padding = torch.zeros(target_rank - current_rank, B_orig.shape[1], 
                                                          dtype=B_orig.dtype, device=B_orig.device)
                                    normalized_params[param_name] = torch.cat([A_orig, A_padding], dim=1)
                                    normalized_params[b_name] = torch.cat([B_orig, B_padding], dim=0)
                            else:
                                # 模式2: A[rank, hidden], B[hidden, rank]
                                if current_rank > target_rank:
                                    normalized_params[param_name] = A_orig[:target_rank, :]
                                    normalized_params[b_name] = B_orig[:, :target_rank]
                                else:
                                    # 填充
                                    A_padding = torch.zeros(target_rank - current_rank, A_orig.shape[1], 
                                                          dtype=A_orig.dtype, device=A_orig.device)
                                    B_padding = torch.zeros(B_orig.shape[0], target_rank - current_rank, 
                                                          dtype=B_orig.dtype, device=B_orig.device)
                                    normalized_params[param_name] = torch.cat([A_orig, A_padding], dim=0)
                                    normalized_params[b_name] = torch.cat([B_orig, B_padding], dim=1)
                    
                    processed_pairs.add(param_name)
                    processed_pairs.add(b_name)
                else:
                    # 没有对应的B参数，直接处理A
                    if param_tensor.shape[1] != target_rank:
                        if param_tensor.shape[1] > target_rank:
                            normalized_params[param_name] = param_tensor[:, :target_rank]
                        else:
                            padding = torch.zeros(param_tensor.shape[0], target_rank - param_tensor.shape[1], 
                                               dtype=param_tensor.dtype, device=param_tensor.device)
                            normalized_params[param_name] = torch.cat([param_tensor, padding], dim=1)
                    else:
                        normalized_params[param_name] = param_tensor
                    processed_pairs.add(param_name)
                    
        # 处理剩余的lora_B参数
        for param_name, param_tensor in lora_params.items():
            if param_name not in processed_pairs and 'lora_B' in param_name:
                if param_tensor.shape[0] != target_rank:
                    if param_tensor.shape[0] > target_rank:
                        normalized_params[param_name] = param_tensor[:target_rank, :]
                    else:
                        padding = torch.zeros(target_rank - param_tensor.shape[0], param_tensor.shape[1], 
                                           dtype=param_tensor.dtype, device=param_tensor.device)
                        normalized_params[param_name] = torch.cat([param_tensor, padding], dim=0)
                else:
                    normalized_params[param_name] = param_tensor
                processed_pairs.add(param_name)
                    
        # 处理其他参数
        for param_name, param_tensor in lora_params.items():
            if param_name not in processed_pairs:
                normalized_params[param_name] = param_tensor
        
        return normalized_params
    
    def _compute_effective_rank(self, lora_params: Dict[str, torch.Tensor]) -> int:
        """
        计算LoRA参数的有效秩
        
        Args:
            lora_params: LoRA参数字典
            
        Returns:
            int: 有效秩
        """
        ranks = []
        for param_name, param_tensor in lora_params.items():
            if 'lora_A' in param_name:
                ranks.append(param_tensor.shape[1])
            elif 'lora_B' in param_name:
                ranks.append(param_tensor.shape[0])
        
        if not ranks:
            return 16  # 默认秩
        
        # 返回最常见的秩
        from collections import Counter
        rank_counts = Counter(ranks)
        return rank_counts.most_common(1)[0][0]
    
    def _validate_lora_dimensions(self, lora_params: Dict[str, torch.Tensor]) -> bool:
        """
        验证LoRA参数的维度一致性
        
        Args:
            lora_params: LoRA参数字典
            
        Returns:
            bool: 维度是否一致
        """
        try:
            # 收集所有lora_A和lora_B的维度信息
            a_dims = {}
            b_dims = {}
            
            for param_name, param_tensor in lora_params.items():
                if 'lora_A' in param_name:
                    base_name = param_name.replace('.lora_A', '').replace('lora_A', '')
                    a_dims[base_name] = param_tensor.shape
                elif 'lora_B' in param_name:
                    base_name = param_name.replace('.lora_B', '').replace('lora_B', '')
                    b_dims[base_name] = param_tensor.shape
            
            # 检查A和B的维度是否匹配
            for base_name in a_dims:
                if base_name in b_dims:
                    a_shape = a_dims[base_name]
                    b_shape = b_dims[base_name]
                    
                    # 根据实际的LoRA实现，检查维度匹配
                    # 通常lora_A是[rank, hidden_size]，lora_B是[hidden_size, rank]
                    # 或者lora_A是[hidden_size, rank]，lora_B是[rank, hidden_size]
                    
                    # 检查两种可能的匹配方式
                    match1 = (a_shape[1] == b_shape[0])  # A[rank, hidden] 和 B[hidden, rank]
                    match2 = (a_shape[0] == b_shape[1])  # A[hidden, rank] 和 B[rank, hidden]
                    
                    if not (match1 or match2):
                        logger.error(f"LoRA维度不匹配: {base_name}")
                        logger.error(f"  lora_A形状: {a_shape}")
                        logger.error(f"  lora_B形状: {b_shape}")
                        logger.error(f"  尝试匹配1 (A[1]==B[0]): {a_shape[1]} == {b_shape[0]} -> {match1}")
                        logger.error(f"  尝试匹配2 (A[0]==B[1]): {a_shape[0]} == {b_shape[1]} -> {match2}")
                        return False
                    else:
                        logger.debug(f"LoRA维度匹配: {base_name} - A{a_shape} B{b_shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"验证LoRA维度时出错: {e}")
            return False
    
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
                # 形状不同，需要特殊处理
                logger.warning(f"参数 {param_name} 形状不一致，跳过融合")
                # 选择权重最大的参数
                max_weight_idx = valid_weights.index(max(valid_weights))
                merged_params[param_name] = param_values[max_weight_idx]
        
        return merged_params
    
    def merge_loras_cross_rank(self, lora_names: List[str], merge_weights: List[float] = None, 
                              target_rank: int = None, merge_strategy: str = "svd") -> PeftModel:
        """
        跨秩融合多个LoRA适配器
        
        Args:
            lora_names: LoRA适配器名称列表
            merge_weights: 融合权重列表，如果为None则使用均等权重
            target_rank: 目标秩，如果为None则自动计算
            merge_strategy: 融合策略 ("svd", "max_rank", "min_rank")
            
        Returns:
            PeftModel: 融合后的模型
        """
        if not lora_names:
            raise ValueError("至少需要指定一个LoRA适配器")
        
        if merge_weights is None:
            merge_weights = [1.0 / len(lora_names)] * len(lora_names)
        
        if len(merge_weights) != len(lora_names):
            raise ValueError("融合权重数量必须与LoRA适配器数量相等")
        
        logger.info(f"开始跨秩融合LoRA适配器: {lora_names}")
        logger.info(f"融合权重: {merge_weights}")
        logger.info(f"融合策略: {merge_strategy}")
        
        # 检查所有LoRA适配器是否存在
        for name in lora_names:
            if name not in self.available_loras:
                raise ValueError(f"LoRA适配器 {name} 不存在")
        
        # 加载所有LoRA适配器
        lora_models = []
        lora_params_list = []
        ranks = []
        
        for lora_name in lora_names:
            lora_path = self.available_loras[lora_name]['path']
            lora_model = self.load_lora(lora_path)
            lora_models.append(lora_model)
            
            # 提取LoRA参数
            lora_params = {}
            for name, param in lora_model.named_parameters():
                if 'lora' in name.lower():
                    lora_params[name] = param.data.clone()
            
            # 验证LoRA参数维度
            if not self._validate_lora_dimensions(lora_params):
                logger.warning(f"LoRA适配器 {lora_name} 维度验证失败，但继续处理")
            
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
                target_rank = self._compute_effective_rank(lora_params_list[0])
            else:
                target_rank = max(ranks)  # 默认使用最大秩
        
        logger.info(f"目标融合秩: {target_rank}")
        
        # 归一化所有LoRA参数到目标秩
        normalized_params_list = []
        for i, (lora_params, lora_name) in enumerate(zip(lora_params_list, lora_names)):
            alpha = self.available_loras[lora_name]['lora_alpha']
            normalized_params = self._normalize_lora_weights(lora_params, target_rank, alpha)
            normalized_params_list.append(normalized_params)
        
        # 融合归一化后的参数
        merged_params = self._weighted_merge_loras(normalized_params_list, merge_weights, target_rank)
        
        # 创建融合后的LoRA配置
        first_lora_info = self.available_loras[lora_names[0]]
        merged_config = LoraConfig(
            r=target_rank,
            lora_alpha=first_lora_info['lora_alpha'],
            target_modules=first_lora_info['target_modules'],
            lora_dropout=first_lora_info['lora_dropout'],
            bias=first_lora_info['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 创建新的PEFT模型
        merged_model = get_peft_model(self.base_model, merged_config)
        
        # 将融合后的参数加载到模型中
        self._load_merged_parameters(merged_model, merged_params)
        
        # 确保模型处于正确状态
        merged_model.eval()
        
        logger.info("跨秩LoRA融合完成")
        return merged_model
    
    def _load_merged_parameters(self, merged_model: PeftModel, merged_params: Dict[str, torch.Tensor]):
        """
        将融合后的参数加载到模型中
        
        Args:
            merged_model: 目标模型
            merged_params: 融合后的参数字典
        """
        # 获取模型的状态字典
        state_dict = merged_model.state_dict()
        
        # 更新参数
        for param_name, param_tensor in merged_params.items():
            if param_name in state_dict:
                target_tensor = state_dict[param_name]
                
                # 检查维度是否匹配
                if param_tensor.shape != target_tensor.shape:
                    logger.warning(f"参数维度不匹配: {param_name}")
                    logger.warning(f"  目标形状: {target_tensor.shape}")
                    logger.warning(f"  源形状: {param_tensor.shape}")
                    
                    # 尝试调整维度
                    try:
                        if param_tensor.numel() == target_tensor.numel():
                            # 元素数量相同，重塑
                            adjusted_tensor = param_tensor.reshape(target_tensor.shape)
                            target_tensor.copy_(adjusted_tensor)
                            logger.info(f"重塑参数: {param_name} {param_tensor.shape} -> {target_tensor.shape}")
                        else:
                            # 元素数量不同，使用截断或填充
                            if param_tensor.numel() > target_tensor.numel():
                                # 截断
                                adjusted_tensor = param_tensor.flatten()[:target_tensor.numel()].reshape(target_tensor.shape)
                            else:
                                # 填充
                                flat_target = torch.zeros_like(target_tensor).flatten()
                                flat_target[:param_tensor.numel()] = param_tensor.flatten()
                                adjusted_tensor = flat_target.reshape(target_tensor.shape)
                            
                            target_tensor.copy_(adjusted_tensor)
                            logger.info(f"调整参数: {param_name} {param_tensor.shape} -> {target_tensor.shape}")
                    except Exception as e:
                        logger.error(f"调整参数失败 {param_name}: {e}，跳过")
                        continue
                else:
                    # 维度匹配，直接复制
                    target_tensor.copy_(param_tensor)
                    logger.debug(f"加载参数: {param_name}")
            else:
                logger.warning(f"未找到目标参数: {param_name}")
    
    def validate_merged_model(self, merged_model: PeftModel) -> bool:
        """
        验证融合后的模型
        
        Args:
            merged_model: 融合后的模型
            
        Returns:
            bool: 验证是否通过
        """
        try:
            logger.info("开始验证融合后的模型...")
            
            # 检查模型是否处于正确状态
            if not hasattr(merged_model, 'peft_config'):
                logger.error("融合后的模型缺少PEFT配置")
                return False
            
            # 检查LoRA参数是否有效
            lora_params = 0
            for name, param in merged_model.named_parameters():
                if 'lora' in name.lower():
                    lora_params += 1
                    # 检查参数是否包含NaN或Inf
                    if torch.isnan(param.data).any():
                        logger.error(f"参数 {name} 包含NaN值")
                        return False
                    if torch.isinf(param.data).any():
                        logger.error(f"参数 {name} 包含Inf值")
                        return False
            
            logger.info(f"验证通过，发现 {lora_params} 个LoRA参数")
            return True
            
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return False
    
    def test_merged_model(self, merged_model: PeftModel, test_prompt: str = "你好") -> str:
        """
        测试融合后的模型
        
        Args:
            merged_model: 融合后的模型
            test_prompt: 测试提示词
            
        Returns:
            str: 生成的文本
        """
        try:
            logger.info(f"测试融合后的模型，输入: {test_prompt}")
            
            # 确保模型在正确的设备上
            merged_model = merged_model.to(self.device)
            
            # 编码输入
            inputs = self.tokenizer.encode(test_prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = merged_model.generate(
                    inputs,
                    max_length=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(test_prompt):].strip()
            
            logger.info(f"模型测试成功，输出: {generated_text}")
            return generated_text
            
        except Exception as e:
            logger.error(f"模型测试失败: {e}")
            return ""
    
    def save_merged_model(self, merged_model: PeftModel, output_path: str, 
                         model_name: str = "Qwen3-14B-lora-norm-v1"):
        """
        保存融合后的模型
        
        Args:
            merged_model: 融合后的模型
            output_path: 输出路径
            model_name: 模型名称
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_save_path = output_path / model_name
        model_save_path.mkdir(exist_ok=True)
        
        try:
            logger.info(f"正在保存融合后的模型到: {model_save_path}")
            
            # 验证模型
            if not self.validate_merged_model(merged_model):
                raise ValueError("融合后的模型验证失败")
            
            # 测试模型
            test_output = self.test_merged_model(merged_model)
            if not test_output:
                logger.warning("模型测试失败，但继续保存")
            
            # 保存模型
            merged_model.save_pretrained(str(model_save_path))
            
            # 保存分词器
            self.tokenizer.save_pretrained(str(model_save_path))
            
            # 创建模型配置文件
            model_config = {
                "model_type": "merged_lora_norm",
                "base_model": str(self.base_model_path),
                "merged_loras": list(self.available_loras.keys()),
                "device": self.device,
                "torch_dtype": "float16" if self.device == "cuda" else "float32",
                "test_output": test_output,
                "validation_passed": True,
                "merge_method": "cross_rank_normalization"
            }
            
            config_path = model_save_path / "merge_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"模型保存成功: {model_save_path}")
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    def interactive_select_loras(self) -> List[str]:
        """
        交互式选择LoRA适配器
        
        Returns:
            List[str]: 选择的LoRA适配器名称列表
        """
        if not self.available_loras:
            print("没有可用的LoRA适配器")
            return []
        
        # 按秩分组显示LoRA适配器
        lora_by_rank = {}
        for name, info in self.available_loras.items():
            rank = info['r']
            if rank not in lora_by_rank:
                lora_by_rank[rank] = []
            lora_by_rank[rank].append(name)
        
        print("\n可用的LoRA适配器（按秩分组）:")
        for rank, loras in sorted(lora_by_rank.items()):
            print(f"\n秩 {rank}:")
            for i, lora_name in enumerate(loras, 1):
                info = self.available_loras[lora_name]
                print(f"  {i}. {lora_name} (alpha={info['lora_alpha']}, dropout={info['lora_dropout']})")
        
        # 选择要融合的LoRA适配器（支持跨秩选择）
        print("\n选择要融合的LoRA适配器（支持不同秩）:")
        selected_names = []
        
        while True:
            try:
                choice = input(f"请输入LoRA适配器编号（多个用逗号分隔，输入'done'完成）: ").strip()
                
                if choice.lower() == 'done':
                    if selected_names:
                        break
                    else:
                        print("请至少选择一个LoRA适配器")
                        continue
                
                # 解析选择
                choices = [c.strip() for c in choice.split(',')]
                for c in choices:
                    if c.isdigit():
                        # 在所有LoRA中查找
                        all_loras = []
                        for rank, loras in sorted(lora_by_rank.items()):
                            all_loras.extend(loras)
                        
                        idx = int(c) - 1
                        if 0 <= idx < len(all_loras):
                            lora_name = all_loras[idx]
                            if lora_name not in selected_names:
                                selected_names.append(lora_name)
                                print(f"已选择: {lora_name}")
                            else:
                                print(f"{lora_name} 已经选择过了")
                        else:
                            print(f"无效选择: {c}")
                    else:
                        print(f"无效输入: {c}")
                
            except (ValueError, IndexError):
                print("请输入有效的数字")
        
        return selected_names
    
    def interactive_set_weights(self, lora_names: List[str]) -> List[float]:
        """
        交互式设置融合权重
        
        Args:
            lora_names: LoRA适配器名称列表
            
        Returns:
            List[float]: 融合权重列表
        """
        print(f"\n为 {len(lora_names)} 个LoRA适配器设置融合权重:")
        
        weights = []
        for i, name in enumerate(lora_names, 1):
            while True:
                try:
                    weight_input = input(f"LoRA适配器 {name} 的权重 (默认: 1.0): ").strip()
                    if not weight_input:
                        weight = 1.0
                    else:
                        weight = float(weight_input)
                    weights.append(weight)
                    break
                except ValueError:
                    print("请输入有效的数字")
        
        # 显示权重设置
        print("\n融合权重设置:")
        total_weight = sum(weights)
        for name, weight in zip(lora_names, weights):
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            print(f"  {name}: {weight} ({percentage:.1f}%)")
        
        return weights


def main():
    """主函数，演示跨秩LoRA融合功能"""
    # 设置路径
    base_model_path = "./Qwen3-14B"
    lora_dir = "."
    output_dir = "./merged_models"
    
    print("跨秩LoRA矩阵融合工具")
    print("="*50)
    
    try:
        # 创建LoRA归一化融合器
        print("正在初始化LoRA归一化融合器...")
        merger = LoRANormMerger(base_model_path)
        
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
        
        # 交互式选择LoRA适配器
        selected_loras = merger.interactive_select_loras()
        
        if not selected_loras:
            print("没有选择LoRA适配器，程序退出")
            return
        
        print(f"\n选择的LoRA适配器: {selected_loras}")
        
        # 设置融合权重
        weights = merger.interactive_set_weights(selected_loras)
        
        # 选择融合策略
        print("\n请选择融合策略:")
        print("1. SVD分解策略（推荐）")
        print("2. 最大秩策略")
        print("3. 最小秩策略")
        
        while True:
            try:
                strategy_choice = input("请输入选择 (1-3): ").strip()
                if strategy_choice in ['1', '2', '3']:
                    break
                print("请输入1、2或3")
            except KeyboardInterrupt:
                print("\n操作已取消")
                return
        
        strategy_map = {
            '1': 'svd',
            '2': 'max_rank',
            '3': 'min_rank'
        }
        merge_strategy = strategy_map[strategy_choice]
        
        # 设置目标秩（可选）
        target_rank = None
        rank_input = input("\n请输入目标秩（直接回车使用自动计算）: ").strip()
        if rank_input.isdigit():
            target_rank = int(rank_input)
        
        # 融合LoRA适配器
        print("\n开始跨秩融合LoRA适配器...")
        merged_model = merger.merge_loras_cross_rank(
            selected_loras, 
            weights, 
            target_rank, 
            merge_strategy
        )
        
        # 保存融合后的模型
        print("正在保存融合后的模型...")
        output_path = merger.save_merged_model(merged_model, output_dir)
        
        print(f"\n跨秩融合完成！模型已保存到: {output_path}")
        
        # 测试融合后的模型
        print("\n测试融合后的模型...")
        test_prompt = "请解释地理加权回归的基本原理："
        print(f"输入: {test_prompt}")
        
        # 生成文本
        inputs = merger.tokenizer.encode(test_prompt, return_tensors="pt").to(merger.device)
        with torch.no_grad():
            outputs = merged_model.generate(
                inputs,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=merger.tokenizer.eos_token_id
            )
        
        generated_text = merger.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(test_prompt):].strip()
        print(f"输出: {generated_text}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

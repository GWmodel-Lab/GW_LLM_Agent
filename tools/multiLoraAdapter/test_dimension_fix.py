#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试维度修复的简单脚本
"""

import torch
import numpy as np
from loraNormMerge import LoRANormMerger

def test_svd_decompose():
    """测试SVD分解方法"""
    print("测试SVD分解方法...")
    
    # 创建一个简单的LoRA归一化融合器（不需要实际加载模型）
    class TestMerger:
        def _svd_decompose(self, matrix: torch.Tensor, target_rank: int):
            """使用SVD分解矩阵到目标秩"""
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
                    print(f"目标秩 {target_rank} 超过矩阵最小维度 {min_dim}，调整为 {actual_target_rank}")
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
                print(f"SVD分解失败: {e}")
                print(f"矩阵形状: {matrix.shape}, 目标秩: {target_rank}")
                raise
    
    merger = TestMerger()
    
    # 测试用例1：正常的LoRA矩阵
    print("\n测试用例1：正常LoRA矩阵")
    lora_A = torch.randn(5120, 8)  # 模拟lora_A参数
    lora_B = torch.randn(8, 5120)  # 模拟lora_B参数
    
    print(f"原始lora_A形状: {lora_A.shape}")
    print(f"原始lora_B形状: {lora_B.shape}")
    
    # 重建完整矩阵
    full_matrix = lora_A @ lora_B
    print(f"完整矩阵形状: {full_matrix.shape}")
    
    # 测试SVD分解到不同秩
    for target_rank in [4, 8, 16]:
        try:
            A_new, B_new = merger._svd_decompose(full_matrix, target_rank)
            print(f"目标秩 {target_rank}: A_new形状={A_new.shape}, B_new形状={B_new.shape}")
            
            # 验证重建
            reconstructed = A_new @ B_new
            print(f"重建矩阵形状: {reconstructed.shape}")
            print(f"重建误差: {torch.norm(full_matrix - reconstructed).item():.6f}")
            
        except Exception as e:
            print(f"目标秩 {target_rank} 失败: {e}")
    
    # 测试用例2：维度不匹配的情况
    print("\n测试用例2：维度不匹配情况")
    try:
        # 创建一个不规则的矩阵
        irregular_matrix = torch.randn(100, 50)
        A_new, B_new = merger._svd_decompose(irregular_matrix, 30)
        print(f"不规则矩阵 {irregular_matrix.shape} -> A: {A_new.shape}, B: {B_new.shape}")
    except Exception as e:
        print(f"不规则矩阵测试失败: {e}")
    
    # 测试用例3：目标秩超过矩阵维度
    print("\n测试用例3：目标秩超过矩阵维度")
    try:
        small_matrix = torch.randn(10, 5)
        A_new, B_new = merger._svd_decompose(small_matrix, 20)  # 目标秩超过最小维度
        print(f"小矩阵 {small_matrix.shape} -> A: {A_new.shape}, B: {B_new.shape}")
    except Exception as e:
        print(f"小矩阵测试失败: {e}")

def test_lora_normalization():
    """测试LoRA权重归一化"""
    print("\n测试LoRA权重归一化...")
    
    # 模拟LoRA参数字典
    lora_params = {
        'base_model.model.layers.0.self_attn.q_proj.lora_A': torch.randn(5120, 8),
        'base_model.model.layers.0.self_attn.q_proj.lora_B': torch.randn(8, 5120),
        'base_model.model.layers.0.self_attn.k_proj.lora_A': torch.randn(5120, 16),
        'base_model.model.layers.0.self_attn.k_proj.lora_B': torch.randn(16, 5120),
    }
    
    print("原始LoRA参数:")
    for name, param in lora_params.items():
        print(f"  {name}: {param.shape}")
    
    # 创建一个简化的归一化方法
    def normalize_lora_weights(lora_params, target_rank):
        normalized_params = {}
        
        # 首先处理所有lora_A参数
        for param_name, param_tensor in lora_params.items():
            if 'lora_A' in param_name:
                if param_tensor.shape[1] != target_rank:
                    print(f"调整 {param_name} 从秩 {param_tensor.shape[1]} 到 {target_rank}")
                    # 简单的截断或填充
                    if param_tensor.shape[1] > target_rank:
                        normalized_params[param_name] = param_tensor[:, :target_rank]
                    else:
                        padding = torch.zeros(param_tensor.shape[0], target_rank - param_tensor.shape[1])
                        normalized_params[param_name] = torch.cat([param_tensor, padding], dim=1)
                else:
                    normalized_params[param_name] = param_tensor
        
        # 然后处理lora_B参数
        for param_name, param_tensor in lora_params.items():
            if 'lora_B' in param_name and param_name not in normalized_params:
                if param_tensor.shape[0] != target_rank:
                    print(f"调整 {param_name} 从秩 {param_tensor.shape[0]} 到 {target_rank}")
                    if param_tensor.shape[0] > target_rank:
                        normalized_params[param_name] = param_tensor[:target_rank, :]
                    else:
                        padding = torch.zeros(target_rank - param_tensor.shape[0], param_tensor.shape[1])
                        normalized_params[param_name] = torch.cat([param_tensor, padding], dim=0)
                else:
                    normalized_params[param_name] = param_tensor
        
        return normalized_params
    
    # 测试不同目标秩
    for target_rank in [4, 8, 16]:
        print(f"\n目标秩: {target_rank}")
        normalized = normalize_lora_weights(lora_params, target_rank)
        
        print("归一化后的参数:")
        for name, param in normalized.items():
            print(f"  {name}: {param.shape}")

if __name__ == "__main__":
    print("开始测试维度修复...")
    
    try:
        test_svd_decompose()
        test_lora_normalization()
        print("\n所有测试完成！")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

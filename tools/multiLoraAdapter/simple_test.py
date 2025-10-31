#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的维度修复测试
"""

import torch

def test_svd_fix():
    """测试SVD分解修复"""
    print("测试SVD分解修复...")
    
    # 模拟原始错误情况：5120和8的维度不匹配
    print("模拟原始错误情况:")
    print("张量a形状: (5120,)")
    print("张量b形状: (8,)")
    
    # 创建一个测试矩阵
    matrix = torch.randn(5120, 8)
    print(f"测试矩阵形状: {matrix.shape}")
    
    # 测试SVD分解
    try:
        U, S, V = torch.svd(matrix)
        print(f"SVD结果: U={U.shape}, S={S.shape}, V={V.shape}")
        
        # 测试目标秩
        target_rank = 4
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        V_truncated = V[:, :target_rank]
        
        print(f"截断后: U_truncated={U_truncated.shape}, S_truncated={S_truncated.shape}, V_truncated={V_truncated.shape}")
        
        # 重新组合
        sqrt_S = torch.sqrt(torch.clamp(S_truncated, min=1e-8))
        A = U_truncated @ torch.diag(sqrt_S)
        B = torch.diag(sqrt_S) @ V_truncated.T
        
        print(f"重新组合: A={A.shape}, B={B.shape}")
        
        # 验证重建
        reconstructed = A @ B
        print(f"重建矩阵形状: {reconstructed.shape}")
        print(f"重建误差: {torch.norm(matrix - reconstructed).item():.6f}")
        
        print("✅ SVD分解测试通过！")
        
    except Exception as e:
        print(f"❌ SVD分解测试失败: {e}")

def test_dimension_validation():
    """测试维度验证"""
    print("\n测试维度验证...")
    
    # 模拟LoRA参数
    lora_params = {
        'lora_A': torch.randn(5120, 8),
        'lora_B': torch.randn(8, 5120),
    }
    
    print("LoRA参数:")
    for name, param in lora_params.items():
        print(f"  {name}: {param.shape}")
    
    # 检查维度匹配
    a_shape = lora_params['lora_A'].shape
    b_shape = lora_params['lora_B'].shape
    
    if a_shape[1] == b_shape[0]:
        print("✅ LoRA维度匹配")
    else:
        print(f"❌ LoRA维度不匹配: A[1]={a_shape[1]}, B[0]={b_shape[0]}")
    
    # 测试目标秩调整
    target_rank = 4
    print(f"\n调整到目标秩: {target_rank}")
    
    # 调整A
    if a_shape[1] != target_rank:
        if a_shape[1] > target_rank:
            A_new = lora_params['lora_A'][:, :target_rank]
        else:
            padding = torch.zeros(a_shape[0], target_rank - a_shape[1])
            A_new = torch.cat([lora_params['lora_A'], padding], dim=1)
        print(f"A调整后: {A_new.shape}")
    else:
        A_new = lora_params['lora_A']
        print(f"A无需调整: {A_new.shape}")
    
    # 调整B
    if b_shape[0] != target_rank:
        if b_shape[0] > target_rank:
            B_new = lora_params['lora_B'][:target_rank, :]
        else:
            padding = torch.zeros(target_rank - b_shape[0], b_shape[1])
            B_new = torch.cat([lora_params['lora_B'], padding], dim=0)
        print(f"B调整后: {B_new.shape}")
    else:
        B_new = lora_params['lora_B']
        print(f"B无需调整: {B_new.shape}")
    
    # 验证调整后的维度
    if A_new.shape[1] == B_new.shape[0]:
        print("✅ 调整后维度匹配")
    else:
        print(f"❌ 调整后维度仍不匹配: A[1]={A_new.shape[1]}, B[0]={B_new.shape[0]}")

if __name__ == "__main__":
    print("开始简单测试...")
    test_svd_fix()
    test_dimension_validation()
    print("\n测试完成！")

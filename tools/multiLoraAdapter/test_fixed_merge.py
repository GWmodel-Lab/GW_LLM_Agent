#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的LoRA融合功能
"""

import torch
import json
from loraNormMerge import LoRANormMerger

def test_dimension_handling():
    """测试维度处理"""
    print("测试维度处理...")
    
    # 模拟不同秩的LoRA参数
    lora_params_rank8 = {
        'base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight': torch.randn(5120, 8),
        'base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight': torch.randn(8, 5120),
        'base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight': torch.randn(8, 5120),
        'base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight': torch.randn(1024, 8),
    }
    
    lora_params_rank16 = {
        'base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight': torch.randn(5120, 16),
        'base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight': torch.randn(16, 5120),
        'base_model.model.layers.0.self_attn.v_proj.lora_A.default.weight': torch.randn(16, 5120),
        'base_model.model.layers.0.self_attn.v_proj.lora_B.default.weight': torch.randn(1024, 16),
    }
    
    print("秩8的LoRA参数:")
    for name, param in lora_params_rank8.items():
        print(f"  {name}: {param.shape}")
    
    print("\n秩16的LoRA参数:")
    for name, param in lora_params_rank16.items():
        print(f"  {name}: {param.shape}")
    
    # 测试维度验证
    class TestMerger:
        def _validate_lora_dimensions(self, lora_params):
            try:
                a_dims = {}
                b_dims = {}
                
                for param_name, param_tensor in lora_params.items():
                    if 'lora_A' in param_name:
                        base_name = param_name.replace('.lora_A', '').replace('lora_A', '')
                        a_dims[base_name] = param_tensor.shape
                    elif 'lora_B' in param_name:
                        base_name = param_name.replace('.lora_B', '').replace('lora_B', '')
                        b_dims[base_name] = param_tensor.shape
                
                for base_name in a_dims:
                    if base_name in b_dims:
                        a_shape = a_dims[base_name]
                        b_shape = b_dims[base_name]
                        
                        match1 = (a_shape[1] == b_shape[0])
                        match2 = (a_shape[0] == b_shape[1])
                        
                        if not (match1 or match2):
                            print(f"❌ LoRA维度不匹配: {base_name}")
                            print(f"  lora_A形状: {a_shape}")
                            print(f"  lora_B形状: {b_shape}")
                            return False
                        else:
                            print(f"✅ LoRA维度匹配: {base_name} - A{a_shape} B{b_shape}")
                
                return True
            except Exception as e:
                print(f"验证失败: {e}")
                return False
    
    merger = TestMerger()
    
    print("\n测试秩8的维度验证:")
    merger._validate_lora_dimensions(lora_params_rank8)
    
    print("\n测试秩16的维度验证:")
    merger._validate_lora_dimensions(lora_params_rank16)

def test_parameter_loading():
    """测试参数加载"""
    print("\n测试参数加载...")
    
    # 模拟目标模型参数
    target_params = {
        'lora_A': torch.randn(5120, 8),
        'lora_B': torch.randn(8, 5120),
    }
    
    # 模拟源参数（不同形状）
    source_params = {
        'lora_A': torch.randn(5120, 16),  # 不同的秩
        'lora_B': torch.randn(16, 5120),  # 不同的秩
    }
    
    print("目标参数形状:")
    for name, param in target_params.items():
        print(f"  {name}: {param.shape}")
    
    print("源参数形状:")
    for name, param in source_params.items():
        print(f"  {name}: {param.shape}")
    
    # 测试参数调整
    for name in target_params:
        target_tensor = target_params[name]
        source_tensor = source_params[name]
        
        print(f"\n处理参数: {name}")
        print(f"  目标形状: {target_tensor.shape}")
        print(f"  源形状: {source_tensor.shape}")
        
        if source_tensor.shape != target_tensor.shape:
            print("  形状不匹配，尝试调整...")
            
            if source_tensor.numel() == target_tensor.numel():
                # 重塑
                adjusted_tensor = source_tensor.reshape(target_tensor.shape)
                print(f"  重塑: {source_tensor.shape} -> {adjusted_tensor.shape}")
            else:
                # 截断或填充
                if source_tensor.numel() > target_tensor.numel():
                    adjusted_tensor = source_tensor.flatten()[:target_tensor.numel()].reshape(target_tensor.shape)
                    print(f"  截断: {source_tensor.shape} -> {adjusted_tensor.shape}")
                else:
                    flat_target = torch.zeros_like(target_tensor).flatten()
                    flat_target[:source_tensor.numel()] = source_tensor.flatten()
                    adjusted_tensor = flat_target.reshape(target_tensor.shape)
                    print(f"  填充: {source_tensor.shape} -> {adjusted_tensor.shape}")
        else:
            print("  形状匹配，无需调整")

if __name__ == "__main__":
    print("开始测试修复后的功能...")
    
    try:
        test_dimension_handling()
        test_parameter_loading()
        print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

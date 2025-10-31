#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断LoRA参数形状问题的脚本
"""

import torch
import json
from pathlib import Path

def analyze_lora_shapes(lora_path: str):
    """分析LoRA适配器的参数形状"""
    print(f"分析LoRA适配器: {lora_path}")
    
    # 加载LoRA配置
    config_path = Path(lora_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"LoRA配置: r={config.get('r')}, alpha={config.get('lora_alpha')}")
    
    # 加载LoRA权重
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 这里需要基础模型，我们只分析权重文件
        print("注意：需要基础模型来完整加载LoRA，这里只分析配置文件")
        
    except ImportError:
        print("PEFT库未安装，无法加载LoRA模型")
        return
    
    # 分析adapter_model.safetensors文件
    safetensors_path = Path(lora_path) / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors import safe_open
            
            print(f"\n分析权重文件: {safetensors_path}")
            
            # 收集所有LoRA参数
            lora_a_params = {}
            lora_b_params = {}
            
            with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    
                    if 'lora_A' in key:
                        lora_a_params[key] = tensor.shape
                    elif 'lora_B' in key:
                        lora_b_params[key] = tensor.shape
            
            print(f"\n发现 {len(lora_a_params)} 个lora_A参数:")
            for name, shape in list(lora_a_params.items())[:5]:  # 只显示前5个
                print(f"  {name}: {shape}")
            if len(lora_a_params) > 5:
                print(f"  ... 还有 {len(lora_a_params) - 5} 个")
            
            print(f"\n发现 {len(lora_b_params)} 个lora_B参数:")
            for name, shape in list(lora_b_params.items())[:5]:  # 只显示前5个
                print(f"  {name}: {shape}")
            if len(lora_b_params) > 5:
                print(f"  ... 还有 {len(lora_b_params) - 5} 个")
            
            # 分析维度模式
            print(f"\n维度模式分析:")
            
            # 按层分组分析
            layer_groups = {}
            for name in lora_a_params:
                if 'layers.' in name:
                    layer_num = name.split('layers.')[1].split('.')[0]
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = {'lora_A': {}, 'lora_B': {}}
                    layer_groups[layer_num]['lora_A'][name] = lora_a_params[name]
            
            for name in lora_b_params:
                if 'layers.' in name:
                    layer_num = name.split('layers.')[1].split('.')[0]
                    if layer_num not in layer_groups:
                        layer_groups[layer_num] = {'lora_A': {}, 'lora_B': {}}
                    layer_groups[layer_num]['lora_B'][name] = lora_b_params[name]
            
            # 分析前几层的维度模式
            for layer_num in sorted(list(layer_groups.keys())[:3]):
                print(f"\n第 {layer_num} 层:")
                layer_data = layer_groups[layer_num]
                
                for a_name, a_shape in layer_data['lora_A'].items():
                    print(f"  {a_name}: {a_shape}")
                    # 找到对应的B参数
                    b_name = a_name.replace('lora_A', 'lora_B')
                    if b_name in layer_data['lora_B']:
                        b_shape = layer_data['lora_B'][b_name]
                        print(f"  {b_name}: {b_shape}")
                        
                        # 检查维度匹配
                        if a_shape[1] == b_shape[0]:
                            print(f"    ✅ 模式1匹配: A[?, {a_shape[1]}] @ B[{b_shape[0]}, ?]")
                        elif a_shape[0] == b_shape[1]:
                            print(f"    ✅ 模式2匹配: A[{a_shape[0]}, ?] @ B[?, {b_shape[1]}]")
                        else:
                            print(f"    ❌ 维度不匹配: A{a_shape} vs B{b_shape}")
                    else:
                        print(f"    ⚠️  未找到对应的B参数")
            
        except ImportError:
            print("safetensors库未安装，无法分析权重文件")
        except Exception as e:
            print(f"分析权重文件时出错: {e}")

def main():
    """主函数"""
    print("LoRA参数形状诊断工具")
    print("=" * 50)
    
    # 分析多个LoRA适配器
    lora_dirs = [
        "qwen3-14b-lora-spatial-stats-final-v1",
        "qwen3-14b-lora-spatial-stats-final-v2", 
        "qwen3-14b-lora-spatial-stats-final-v3",
        "qwen3-14b-lora-spatial-stats-final-16Rank"
    ]
    
    for lora_dir in lora_dirs:
        if Path(lora_dir).exists():
            analyze_lora_shapes(lora_dir)
            print("\n" + "=" * 50)
        else:
            print(f"LoRA目录不存在: {lora_dir}")

if __name__ == "__main__":
    main()

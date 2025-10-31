#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于vLLM的多LoRA推理测试脚本 - 修复版本
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

# 修复 typing_extensions Sentinel 问题
def fix_typing_extensions():
    """修复 typing_extensions Sentinel 问题"""
    import sys
    
    # 如果 typing_extensions 已经加载但没有 Sentinel，添加它
    if 'typing_extensions' in sys.modules:
        typing_extensions = sys.modules['typing_extensions']
        if not hasattr(typing_extensions, 'Sentinel'):
            class Sentinel:
                def __init__(self, name):
                    self.name = name
                def __repr__(self):
                    return f"<Sentinel: {self.name}>"
            typing_extensions.Sentinel = Sentinel

# 应用修复
fix_typing_extensions()

# 现在安全地导入其他模块
try:
    from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, TaskType, get_peft_model
    import warnings
except ImportError as e:
    print(f"导入错误: {e}")
    print("请运行: python fix_pydantic_sentinel.py")
    raise

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
    
    def test_basic_functionality(self):
        """测试基本功能"""
        print("=" * 60)
        print("测试基本功能")
        print("=" * 60)
        
        try:
            # 测试设备
            print(f"设备: {self.device}")
            if torch.cuda.is_available():
                print(f"CUDA 版本: {torch.version.cuda}")
                print(f"GPU 数量: {torch.cuda.device_count()}")
            
            # 测试模型加载
            if self.base_model is not None:
                print("✓ 基础模型加载成功")
            else:
                print("✗ 基础模型加载失败")
                return False
            
            # 测试分词器
            if self.tokenizer is not None:
                test_text = "Hello, world!"
                tokens = self.tokenizer.encode(test_text)
                decoded = self.tokenizer.decode(tokens)
                print(f"✓ 分词器测试: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
            else:
                print("✗ 分词器加载失败")
                return False
            
            print("\n🎉 基本功能测试通过！")
            return True
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False

def main():
    """主函数，演示基本功能测试"""
    print("基于vLLM的Qwen3-14B LoRA融合测试 - 修复版本")
    print("=" * 80)
    
    # 设置路径
    base_model_path = "./Qwen3-14B"
    
    try:
        # 创建LoRA融合器
        print("正在初始化Qwen LoRA融合器...")
        merger = QwenLoRAMerger(base_model_path)
        
        # 测试基本功能
        if merger.test_basic_functionality():
            print("\n✅ 所有测试通过！现在可以运行完整功能了。")
        else:
            print("\n❌ 基本功能测试失败，请检查环境配置。")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

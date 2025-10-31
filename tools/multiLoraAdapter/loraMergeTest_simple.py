#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的 LoRA 融合测试脚本
专注于解决导入问题并提供基本功能
"""

import os
import json
import logging
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import torch
import numpy as np

# 修复 typing_extensions 问题
import sys
if 'typing_extensions' in sys.modules:
    typing_extensions = sys.modules['typing_extensions']
    if not hasattr(typing_extensions, 'Sentinel'):
        class Sentinel:
            def __init__(self, name):
                self.name = name
            def __repr__(self):
                return f"<Sentinel: {self.name}>"
        typing_extensions.Sentinel = Sentinel

# 尝试导入 vLLM，如果失败则使用替代方案
VLLM_AVAILABLE = False
try:
    # 尝试多种导入方式
    try:
        from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
        from vllm.lora.request import LoRARequest
        VLLM_AVAILABLE = True
        print("✓ 使用标准 vLLM 导入")
    except ImportError:
        try:
            from vllm.engine.arg_utils import EngineArgs
            from vllm.engine.llm_engine import LLMEngine
            from vllm.outputs import RequestOutput
            from vllm.sampling_params import SamplingParams
            from vllm.lora.request import LoRARequest
            VLLM_AVAILABLE = True
            print("✓ 使用子模块 vLLM 导入")
        except ImportError:
            print("⚠ vLLM 导入失败，将使用基础功能")
except Exception as e:
    print(f"⚠ vLLM 导入错误: {e}")

# 导入其他必要的库
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLoRAMerger:
    """
    简化的 LoRA 融合器
    专注于基本功能，不依赖 vLLM
    """
    
    def __init__(self, base_model_path: str, device: str = "auto"):
        """
        初始化 LoRA 融合器
        
        Args:
            base_model_path: 基础模型路径
            device: 设备类型
        """
        self.base_model_path = Path(base_model_path)
        self.device = self._get_device(device)
        self.base_model = None
        self.tokenizer = None
        self.available_loras = {}
        
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
            
            # 测试 vLLM 可用性
            if VLLM_AVAILABLE:
                print("✓ vLLM 可用，支持高级功能")
            else:
                print("⚠ vLLM 不可用，仅支持基础功能")
            
            print("\n🎉 基本功能测试通过！")
            return True
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False
    
    def simple_generation_test(self, prompt: str = "请解释地理加权回归的基本原理："):
        """简单的生成测试"""
        if self.base_model is None or self.tokenizer is None:
            print("模型或分词器未加载")
            return None
        
        try:
            print(f"测试提示: {prompt}")
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            print(f"生成结果: {response}")
            return response
            
        except Exception as e:
            print(f"生成测试失败: {e}")
            return None

def main():
    """主函数"""
    print("简化的 LoRA 融合测试")
    print("=" * 80)
    
    # 设置路径
    base_model_path = "./Qwen3-14B"
    
    try:
        # 创建 LoRA 融合器
        print("正在初始化 LoRA 融合器...")
        merger = SimpleLoRAMerger(base_model_path)
        
        # 测试基本功能
        if merger.test_basic_functionality():
            print("\n✅ 基本功能测试通过！")
            
            # 进行简单的生成测试
            print("\n进行生成测试...")
            merger.simple_generation_test()
            
            # 扫描 LoRA 适配器
            print("\n扫描 LoRA 适配器...")
            lora_info = merger.scan_loras(".")
            if lora_info:
                print(f"发现 {len(lora_info)} 个 LoRA 适配器:")
                for name, info in lora_info.items():
                    print(f"  - {name}: 秩={info['r']}, alpha={info['lora_alpha']}")
            else:
                print("未发现 LoRA 适配器")
        else:
            print("\n❌ 基本功能测试失败")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

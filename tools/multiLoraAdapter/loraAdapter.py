#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA适配器脚本
用于加载和切换多个不同的LoRA矩阵，支持地理加权回归、空间分析等领域的微调模型
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRAAdapter:
    """
    LoRA适配器类，用于管理多个LoRA矩阵的加载和切换
    """
    
    def __init__(self, base_model_path: str, device: str = "auto"):
        """
        初始化LoRA适配器
        
        Args:
            base_model_path: 基础模型路径
            device: 设备类型 ("auto", "cpu", "cuda", "mps")
        """
        self.base_model_path = Path(base_model_path)
        self.device = self._get_device(device)
        self.base_model = None
        self.tokenizer = None
        self.current_lora = None
        self.loaded_loras = {}  # 存储已加载的LoRA适配器
        self.lora_configs = {}  # 存储LoRA配置信息
        
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
    
    def load_lora(self, lora_path: str, lora_name: str = None) -> bool:
        """
        加载LoRA适配器
        
        Args:
            lora_path: LoRA适配器路径
            lora_name: LoRA适配器名称（可选，默认使用路径名）
            
        Returns:
            bool: 加载是否成功
        """
        try:
            lora_path = Path(lora_path)
            if not lora_path.exists():
                logger.error(f"LoRA路径不存在: {lora_path}")
                return False
            
            # 使用路径名作为默认名称
            if lora_name is None:
                lora_name = lora_path.name
            
            # 检查是否已经加载
            if lora_name in self.loaded_loras:
                logger.warning(f"LoRA适配器 {lora_name} 已经加载")
                return True
            
            # 加载LoRA配置
            config_path = lora_path / "adapter_config.json"
            if not config_path.exists():
                logger.error(f"LoRA配置文件不存在: {config_path}")
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                lora_config = json.load(f)
            
            # 更新基础模型路径为绝对路径
            lora_config['base_model_name_or_path'] = str(self.base_model_path.absolute())
            
            # 保存配置
            self.lora_configs[lora_name] = lora_config
            
            # 创建PEFT配置
            peft_config = LoraConfig(
                r=lora_config['r'],
                lora_alpha=lora_config['lora_alpha'],
                target_modules=lora_config['target_modules'],
                lora_dropout=lora_config['lora_dropout'],
                bias=lora_config['bias'],
                task_type=TaskType.CAUSAL_LM,
            )
            
            # 加载LoRA适配器
            lora_model = PeftModel.from_pretrained(
                self.base_model,
                str(lora_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # 存储LoRA适配器
            self.loaded_loras[lora_name] = lora_model
            
            logger.info(f"LoRA适配器 {lora_name} 加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载LoRA适配器失败: {e}")
            return False
    
    def switch_lora(self, lora_name: str) -> bool:
        """
        切换到指定的LoRA适配器
        
        Args:
            lora_name: LoRA适配器名称
            
        Returns:
            bool: 切换是否成功
        """
        try:
            if lora_name not in self.loaded_loras:
                logger.error(f"LoRA适配器 {lora_name} 未加载")
                return False
            
            # 卸载当前LoRA
            if self.current_lora is not None:
                self.unload_current_lora()
            
            # 切换到新的LoRA
            self.current_lora = lora_name
            logger.info(f"已切换到LoRA适配器: {lora_name}")
            return True
            
        except Exception as e:
            logger.error(f"切换LoRA适配器失败: {e}")
            return False
    
    def unload_current_lora(self):
        """卸载当前LoRA适配器"""
        if self.current_lora is not None:
            logger.info(f"卸载当前LoRA适配器: {self.current_lora}")
            self.current_lora = None
    
    def unload_lora(self, lora_name: str) -> bool:
        """
        卸载指定的LoRA适配器
        
        Args:
            lora_name: LoRA适配器名称
            
        Returns:
            bool: 卸载是否成功
        """
        try:
            if lora_name not in self.loaded_loras:
                logger.warning(f"LoRA适配器 {lora_name} 未加载")
                return False
            
            # 如果正在使用该LoRA，先卸载
            if self.current_lora == lora_name:
                self.unload_current_lora()
            
            # 从内存中移除
            del self.loaded_loras[lora_name]
            del self.lora_configs[lora_name]
            
            logger.info(f"LoRA适配器 {lora_name} 卸载成功")
            return True
            
        except Exception as e:
            logger.error(f"卸载LoRA适配器失败: {e}")
            return False
    
    def get_current_model(self):
        """获取当前模型（基础模型+当前LoRA）"""
        if self.current_lora is not None:
            return self.loaded_loras[self.current_lora]
        return self.base_model
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top_p参数
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        try:
            model = self.get_current_model()
            
            # 编码输入
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除输入部分，只返回生成的部分
            generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"文本生成失败: {e}")
            return ""
    
    def get_loaded_loras(self) -> List[str]:
        """获取已加载的LoRA适配器列表"""
        return list(self.loaded_loras.keys())
    
    def get_current_lora(self) -> Optional[str]:
        """获取当前使用的LoRA适配器名称"""
        return self.current_lora
    
    def get_lora_info(self, lora_name: str) -> Optional[Dict]:
        """获取LoRA适配器信息"""
        if lora_name in self.lora_configs:
            return self.lora_configs[lora_name]
        return None
    
    def load_all_loras(self, lora_dir: str) -> Dict[str, bool]:
        """
        加载指定目录下的所有LoRA适配器
        
        Args:
            lora_dir: LoRA适配器目录
            
        Returns:
            Dict[str, bool]: 加载结果字典
        """
        lora_dir = Path(lora_dir)
        results = {}
        
        if not lora_dir.exists():
            logger.error(f"LoRA目录不存在: {lora_dir}")
            return results
        
        # 查找所有LoRA适配器目录
        for item in lora_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                lora_name = item.name
                success = self.load_lora(str(item), lora_name)
                results[lora_name] = success
        
        return results
    
    def find_lora_by_name(self, search_name: str) -> List[str]:
        """
        根据名称搜索LoRA适配器
        
        Args:
            search_name: 搜索名称（支持部分匹配）
            
        Returns:
            List[str]: 匹配的LoRA适配器名称列表
        """
        search_name = search_name.lower()
        matches = []
        
        for lora_name in self.loaded_loras.keys():
            if search_name in lora_name.lower():
                matches.append(lora_name)
        
        return matches
    
    def select_lora_interactive(self) -> Optional[str]:
        """
        交互式选择LoRA适配器
        
        Returns:
            Optional[str]: 选择的LoRA适配器名称，如果取消则返回None
        """
        if not self.loaded_loras:
            print("没有已加载的LoRA适配器")
            return None
        
        print("\n可用的LoRA适配器:")
        lora_list = list(self.loaded_loras.keys())
        for i, lora_name in enumerate(lora_list, 1):
            print(f"{i}. {lora_name}")
        
        while True:
            try:
                choice = input(f"\n请选择LoRA适配器 (1-{len(lora_list)}) 或输入名称进行搜索 (输入'q'退出): ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                # 尝试按数字选择
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(lora_list):
                        return lora_list[choice_num - 1]
                    else:
                        print(f"请输入1到{len(lora_list)}之间的数字")
                        continue
                
                # 按名称搜索
                matches = self.find_lora_by_name(choice)
                if not matches:
                    print(f"未找到包含'{choice}'的LoRA适配器")
                    continue
                elif len(matches) == 1:
                    return matches[0]
                else:
                    print(f"找到多个匹配的LoRA适配器:")
                    for i, match in enumerate(matches, 1):
                        print(f"{i}. {match}")
                    
                    while True:
                        sub_choice = input(f"请选择 (1-{len(matches)}) 或输入'q'返回: ").strip()
                        if sub_choice.lower() == 'q':
                            break
                        if sub_choice.isdigit():
                            sub_choice_num = int(sub_choice)
                            if 1 <= sub_choice_num <= len(matches):
                                return matches[sub_choice_num - 1]
                        print(f"请输入1到{len(matches)}之间的数字")
                    continue
                    
            except KeyboardInterrupt:
                print("\n操作已取消")
                return None
            except Exception as e:
                print(f"输入错误: {e}")
                continue
    
    def switch_lora_by_name(self, lora_name: str) -> bool:
        """
        通过名称切换LoRA适配器（支持部分匹配）
        
        Args:
            lora_name: LoRA适配器名称（支持部分匹配）
            
        Returns:
            bool: 切换是否成功
        """
        # 首先尝试精确匹配
        if lora_name in self.loaded_loras:
            return self.switch_lora(lora_name)
        
        # 如果精确匹配失败，尝试部分匹配
        matches = self.find_lora_by_name(lora_name)
        if not matches:
            logger.error(f"未找到包含'{lora_name}'的LoRA适配器")
            return False
        elif len(matches) == 1:
            logger.info(f"找到匹配的LoRA适配器: {matches[0]}")
            return self.switch_lora(matches[0])
        else:
            logger.error(f"找到多个匹配的LoRA适配器: {matches}，请使用更具体的名称")
            return False


def main():
    """主函数，演示LoRA适配器的使用"""
    # 设置路径
    base_model_path = "./Qwen3-14B"
    lora_dir = "."
    
    # 创建LoRA适配器
    print("正在初始化LoRA适配器...")
    adapter = LoRAAdapter(base_model_path)
    
    # 加载所有LoRA适配器
    print("正在加载LoRA适配器...")
    results = adapter.load_all_loras(lora_dir)
    
    for lora_name, success in results.items():
        status = "成功" if success else "失败"
        print(f"LoRA适配器 {lora_name}: {status}")
    
    # 显示已加载的LoRA适配器
    loaded_loras = adapter.get_loaded_loras()
    print(f"\n已加载的LoRA适配器: {loaded_loras}")
    
    if not loaded_loras:
        print("没有可用的LoRA适配器，程序退出")
        return
    
    # 交互式选择LoRA适配器
    print("\n" + "="*50)
    print("LoRA适配器选择界面")
    print("="*50)
    
    while True:
        try:
            # 显示当前状态
            current_lora = adapter.get_current_lora()
            if current_lora:
                print(f"\n当前使用的LoRA适配器: {current_lora}")
            else:
                print("\n当前使用基础模型（未加载LoRA适配器）")
            
            # 显示菜单
            print("\n请选择操作:")
            print("1. 选择LoRA适配器")
            print("2. 通过名称搜索LoRA适配器")
            print("3. 生成文本")
            print("4. 显示LoRA适配器信息")
            print("5. 退出")
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                # 交互式选择
                selected_lora = adapter.select_lora_interactive()
                if selected_lora:
                    if adapter.switch_lora(selected_lora):
                        print(f"成功切换到LoRA适配器: {selected_lora}")
                    else:
                        print(f"切换失败")
                else:
                    print("未选择LoRA适配器")
            
            elif choice == '2':
                # 通过名称搜索
                search_name = input("请输入LoRA适配器名称（支持部分匹配）: ").strip()
                if search_name:
                    if adapter.switch_lora_by_name(search_name):
                        print(f"成功切换到LoRA适配器")
                    else:
                        print(f"切换失败")
                else:
                    print("请输入有效的名称")
            
            elif choice == '3':
                # 生成文本
                prompt = input("请输入提示词: ").strip()
                if prompt:
                    print("正在生成文本...")
                    generated_text = adapter.generate(prompt, max_length=512)
                    print(f"\n生成的文本:\n{generated_text}")
                else:
                    print("请输入有效的提示词")
            
            elif choice == '4':
                # 显示LoRA适配器信息
                print("\n已加载的LoRA适配器:")
                for lora_name in loaded_loras:
                    info = adapter.get_lora_info(lora_name)
                    if info:
                        print(f"\n名称: {lora_name}")
                        print(f"  r: {info.get('r', 'N/A')}")
                        print(f"  lora_alpha: {info.get('lora_alpha', 'N/A')}")
                        print(f"  target_modules: {info.get('target_modules', 'N/A')}")
                        print(f"  lora_dropout: {info.get('lora_dropout', 'N/A')}")
                        print(f"  bias: {info.get('bias', 'N/A')}")
            
            elif choice == '5':
                print("程序退出")
                break
            
            else:
                print("无效选择，请输入1-5之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue


if __name__ == "__main__":
    main()


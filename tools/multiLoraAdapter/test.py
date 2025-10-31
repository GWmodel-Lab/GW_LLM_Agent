#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA矩阵融合脚本
用于将多个同秩的LoRA矩阵融合到基础模型中，并保存为新的模型
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


class LoRAMerger:
    """
    LoRA矩阵融合类，用于将多个LoRA适配器融合到基础模型中
    """

    def __init__(self, base_model_path: str, device: str = "auto"):
        """
        初始化LoRA融合器

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

                    # 检查是否为同秩LoRA
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

    def find_compatible_loras(self, target_r: int = None) -> List[str]:
        """
        查找兼容的LoRA适配器（同秩）

        Args:
            target_r: 目标秩，如果为None则使用第一个LoRA的秩

        Returns:
            List[str]: 兼容的LoRA适配器名称列表
        """
        if not self.available_loras:
            logger.warning("没有可用的LoRA适配器")
            return []

        # 如果没有指定目标秩，使用第一个LoRA的秩
        if target_r is None:
            first_lora = list(self.available_loras.values())[0]
            target_r = first_lora['r']
            logger.info(f"使用第一个LoRA的秩: {target_r}")

        compatible_loras = []
        for name, info in self.available_loras.items():
            if info['r'] == target_r:
                compatible_loras.append(name)

        logger.info(f"找到 {len(compatible_loras)} 个秩为 {target_r} 的LoRA适配器")
        return compatible_loras

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

    def merge_loras(self, lora_names: List[str], merge_weights: List[float] = None) -> PeftModel:
        """
        融合多个LoRA适配器

        Args:
            lora_names: LoRA适配器名称列表
            merge_weights: 融合权重列表，如果为None则使用均等权重

        Returns:
            PeftModel: 融合后的模型
        """
        if not lora_names:
            raise ValueError("至少需要指定一个LoRA适配器")

        if merge_weights is None:
            merge_weights = [1.0 / len(lora_names)] * len(lora_names)

        if len(merge_weights) != len(lora_names):
            raise ValueError("融合权重数量必须与LoRA适配器数量相等")

        logger.info(f"开始融合LoRA适配器: {lora_names}")
        logger.info(f"融合权重: {merge_weights}")

        # 检查所有LoRA适配器是否兼容
        target_r = self.available_loras[lora_names[0]]['r']
        for name in lora_names:
            if name not in self.available_loras:
                raise ValueError(f"LoRA适配器 {name} 不存在")
            if self.available_loras[name]['r'] != target_r:
                raise ValueError(f"LoRA适配器 {name} 的秩不匹配")

        # 加载第一个LoRA适配器作为基础
        first_lora_path = self.available_loras[lora_names[0]]['path']
        merged_model = self.load_lora(first_lora_path)

        # 获取第一个LoRA的权重
        first_weight = merge_weights[0]
        if first_weight != 1.0:
            # 调整第一个LoRA的权重
            for name, param in merged_model.named_parameters():
                if 'lora' in name.lower():
                    param.data *= first_weight

        # 融合其他LoRA适配器
        for i, lora_name in enumerate(lora_names[1:], 1):
            lora_path = self.available_loras[lora_name]['path']
            lora_model = self.load_lora(lora_path)
            weight = merge_weights[i]

            logger.info(f"正在融合LoRA适配器 {lora_name} (权重: {weight})")

            # 融合权重
            for (name1, param1), (name2, param2) in zip(
                    merged_model.named_parameters(),
                    lora_model.named_parameters()
            ):
                if 'lora' in name1.lower() and 'lora' in name2.lower():
                    if name1 == name2:
                        param1.data += param2.data * weight
                    else:
                        logger.warning(f"参数名称不匹配: {name1} vs {name2}")

        logger.info("LoRA适配器融合完成")
        return merged_model

    def save_merged_model(self, merged_model: PeftModel, output_path: str,
                          model_name: str = "Qwen3-14B-lora-v1"):
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

            # 保存模型
            merged_model.save_pretrained(str(model_save_path))

            # 保存分词器
            self.tokenizer.save_pretrained(str(model_save_path))

            # 创建模型配置文件
            model_config = {
                "model_type": "merged_lora",
                "base_model": str(self.base_model_path),
                "merged_loras": list(self.available_loras.keys()),
                "device": self.device,
                "torch_dtype": "float16" if self.device == "cuda" else "float32"
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

        # 选择秩
        while True:
            try:
                rank_choice = input(f"\n请选择要融合的LoRA适配器的秩 (1-{len(lora_by_rank)}): ").strip()
                if rank_choice.isdigit():
                    rank_idx = int(rank_choice) - 1
                    if 0 <= rank_idx < len(lora_by_rank):
                        selected_rank = list(lora_by_rank.keys())[rank_idx]
                        selected_loras = lora_by_rank[selected_rank]
                        break
                print("请输入有效的数字")
            except (ValueError, IndexError):
                print("请输入有效的数字")

        # 选择具体的LoRA适配器
        print(f"\n秩 {selected_rank} 的LoRA适配器:")
        for i, lora_name in enumerate(selected_loras, 1):
            print(f"{i}. {lora_name}")

        selected_names = []
        while True:
            try:
                choice = input(f"\n请选择LoRA适配器 (输入数字，多个用逗号分隔，输入'done'完成): ").strip()

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
                        idx = int(c) - 1
                        if 0 <= idx < len(selected_loras):
                            lora_name = selected_loras[idx]
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
    """主函数，演示LoRA融合功能"""
    # 设置路径
    base_model_path = "./Qwen3-14B"
    lora_dir = "."
    output_dir = "./merged_models"

    print("LoRA矩阵融合工具")
    print("=" * 50)

    try:
        # 创建LoRA融合器
        print("正在初始化LoRA融合器...")
        merger = LoRAMerger(base_model_path)

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

        # 融合LoRA适配器
        print("\n开始融合LoRA适配器...")
        merged_model = merger.merge_loras(selected_loras, weights)

        # 保存融合后的模型
        print("正在保存融合后的模型...")
        output_path = merger.save_merged_model(merged_model, output_dir)

        print(f"\n融合完成！模型已保存到: {output_path}")

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

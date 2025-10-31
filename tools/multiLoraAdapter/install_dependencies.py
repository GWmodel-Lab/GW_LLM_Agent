#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装脚本
用于安装和更新项目所需的Python包
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"正在{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description}成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description}失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("错误: 需要Python 3.8或更高版本")
        return False
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def install_package(package, description=None):
    """安装单个包"""
    if description is None:
        description = f"安装{package}"
    return run_command(f"pip install {package}", description)

def main():
    """主安装函数"""
    print("=" * 60)
    print("GW_LLM_Agent 依赖安装脚本")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 升级pip
    run_command("python -m pip install --upgrade pip", "升级pip")
    
    # 安装基础依赖
    print("\n1. 安装基础依赖...")
    base_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.64.0",
        "huggingface_hub>=0.16.0",
        "typing_extensions>=4.5.0"
    ]
    
    for package in base_packages:
        install_package(package)
    
    # 安装vLLM（需要特殊处理）
    print("\n2. 安装vLLM...")
    print("注意: vLLM需要CUDA支持，如果遇到问题请参考官方文档")
    
    # 尝试安装vLLM
    if not install_package("vllm>=0.2.0", "安装vLLM"):
        print("vLLM安装失败，尝试安装CPU版本...")
        install_package("vllm-cpu", "安装vLLM CPU版本")
    
    # 安装可选依赖
    print("\n3. 安装可选依赖...")
    optional_packages = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0"
    ]
    
    for package in optional_packages:
        install_package(package)
    
    # 验证安装
    print("\n4. 验证安装...")
    try:
        import torch
        import transformers
        import peft
        import vllm
        import typing_extensions
        print("✓ 所有核心依赖安装成功")
        
        # 检查typing_extensions版本
        version = typing_extensions.__version__
        print(f"✓ typing_extensions版本: {version}")
        
        # 检查Sentinel是否可用
        try:
            from typing_extensions import Sentinel
            print("✓ Sentinel导入成功")
        except ImportError:
            print("⚠ Sentinel不可用，但代码中有兼容性处理")
            
    except ImportError as e:
        print(f"✗ 依赖验证失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

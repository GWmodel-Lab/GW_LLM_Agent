#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 pydantic 和 typing_extensions 的 Sentinel 导入问题
"""

import subprocess
import sys
import os

def check_current_versions():
    """检查当前包版本"""
    print("检查当前包版本...")
    
    try:
        import typing_extensions
        print(f"typing_extensions: {typing_extensions.__version__}")
    except ImportError:
        print("typing_extensions: 未安装")
    
    try:
        import pydantic
        print(f"pydantic: {pydantic.__version__}")
    except ImportError:
        print("pydantic: 未安装")
    
    try:
        import vllm
        print(f"vllm: {vllm.__version__}")
    except ImportError:
        print("vllm: 未安装")

def fix_typing_extensions():
    """修复 typing_extensions 问题"""
    print("\n修复 typing_extensions...")
    
    commands = [
        # 卸载并重新安装 typing_extensions
        "pip uninstall typing_extensions -y",
        "pip install typing_extensions>=4.5.0",
        
        # 确保 pydantic 兼容
        "pip install --upgrade pydantic>=2.0.0",
        
        # 重新安装 vllm
        "pip install --upgrade vllm>=0.2.0"
    ]
    
    for cmd in commands:
        print(f"执行: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✓ 成功")
        except subprocess.CalledProcessError as e:
            print(f"✗ 失败: {e}")
            print(f"错误: {e.stderr}")

def test_imports():
    """测试导入"""
    print("\n测试导入...")
    
    try:
        from typing_extensions import Sentinel
        print("✓ typing_extensions.Sentinel 导入成功")
        
        import pydantic
        print("✓ pydantic 导入成功")
        
        from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
        print("✓ vllm 核心模块导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def create_monkey_patch():
    """创建猴子补丁作为临时解决方案"""
    print("\n创建临时解决方案...")
    
    patch_code = '''
# 临时修复 typing_extensions Sentinel 问题
import sys

# 在导入 vllm 之前应用补丁
if 'typing_extensions' in sys.modules:
    typing_extensions = sys.modules['typing_extensions']
    if not hasattr(typing_extensions, 'Sentinel'):
        class Sentinel:
            def __init__(self, name):
                self.name = name
            def __repr__(self):
                return f"<Sentinel: {self.name}>"
        typing_extensions.Sentinel = Sentinel
'''
    
    with open('typing_extensions_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch_code)
    
    print("✓ 创建了 typing_extensions_patch.py")

def main():
    """主函数"""
    print("=" * 60)
    print("修复 pydantic 和 typing_extensions 的 Sentinel 问题")
    print("=" * 60)
    
    # 检查当前版本
    check_current_versions()
    
    # 尝试修复
    fix_typing_extensions()
    
    # 测试导入
    if test_imports():
        print("\n🎉 修复成功！")
        return True
    else:
        print("\n⚠ 自动修复失败，创建临时解决方案...")
        create_monkey_patch()
        print("\n请在使用 vLLM 之前导入 typing_extensions_patch.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

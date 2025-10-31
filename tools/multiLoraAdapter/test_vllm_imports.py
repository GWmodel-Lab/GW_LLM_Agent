#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 vLLM 0.11.0 的正确导入方式
"""

def test_vllm_imports():
    """测试 vLLM 导入"""
    print("测试 vLLM 0.11.0 导入...")
    
    # 测试1: 直接从 vllm 导入
    try:
        from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
        from vllm.lora.request import LoRARequest
        print("✓ 方法1: 直接从 vllm 导入成功")
        return True
    except ImportError as e:
        print(f"✗ 方法1 失败: {e}")
    
    # 测试2: 从子模块导入
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.engine.llm_engine import LLMEngine
        from vllm.outputs import RequestOutput
        from vllm.sampling_params import SamplingParams
        from vllm.lora.request import LoRARequest
        print("✓ 方法2: 从子模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 方法2 失败: {e}")
    
    # 测试3: 检查可用的模块
    try:
        import vllm
        print(f"vLLM 版本: {vllm.__version__}")
        print("可用的属性:")
        for attr in sorted(dir(vllm)):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    except Exception as e:
        print(f"检查 vllm 模块失败: {e}")
    
    # 测试4: 检查子模块
    try:
        from vllm import engine
        print("engine 模块可用")
        print("engine 子模块:")
        for attr in sorted(dir(engine)):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    except Exception as e:
        print(f"检查 engine 模块失败: {e}")
    
    return False

def test_alternative_imports():
    """测试替代导入方式"""
    print("\n测试替代导入方式...")
    
    # 测试不同的导入路径
    import_paths = [
        ("vllm.engine.arg_utils", "EngineArgs"),
        ("vllm.engine.llm_engine", "LLMEngine"),
        ("vllm.outputs", "RequestOutput"),
        ("vllm.sampling_params", "SamplingParams"),
        ("vllm.lora.request", "LoRARequest"),
    ]
    
    for module_path, class_name in import_paths:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_path}.{class_name} 导入成功")
        except (ImportError, AttributeError) as e:
            print(f"✗ {module_path}.{class_name} 导入失败: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM 0.11.0 导入测试")
    print("=" * 60)
    
    success = test_vllm_imports()
    test_alternative_imports()
    
    if success:
        print("\n🎉 找到正确的导入方式！")
    else:
        print("\n⚠ 需要进一步调试导入问题")

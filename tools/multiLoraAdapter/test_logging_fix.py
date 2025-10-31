#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试日志修复
"""

import logging
import sys

def test_logging():
    """测试日志配置"""
    print("测试日志配置...")
    
    # 测试正确的格式
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("这是一条测试日志消息")
        print("✅ 正确的日志格式工作正常")
    except Exception as e:
        print(f"❌ 正确的日志格式失败: {e}")
    
    # 测试错误的格式（应该会失败）
    try:
        # 清除之前的配置
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("这是一条测试日志消息")
        print("❌ 错误的日志格式应该失败但没有失败")
    except Exception as e:
        print(f"✅ 错误的日志格式正确失败: {e}")

if __name__ == "__main__":
    test_logging()

# 导入必要的库
import json
import os
import time
from openai import OpenAI
from typing import List, Dict


def generate_instruction_with_qwen(question: str, answer: str, api_key: str, max_retries=3) -> str:
    """
    调用阿里云百炼Qwen模型根据问答内容生成主题描述(instruction)

    参数:
        question: 问题文本
        answer: 答案文本
        api_key: 阿里云百炼API密钥
        max_retries: 最大重试次数

    返回:
        生成的instruction文本
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    prompt = f"""请根据以下问答内容生成一个简洁的主题描述(instruction)：

    问题: {question}
    回答: {answer}

    要求:
    1. 用中文描述且不超过10个字
    2. 准确概括问答的核心主题
    3. 只生成主题描述文本，不要包含"主题描述："、"**主题描述**："等前缀
    4. 不要包含任何评估语句、质量判断或其他额外信息
    5. 如果可能，尽量从空间分析或统计建模角度进行概括
    6. 在以下情况下返回"DELETE"标记：
       - 内容完全无关（如纯文学、艺术、生活等）
       - 问题明确要求提及图表、图像或公式但未提供具体内容，如有"方程(12)"、"图3"等但未提供实际方程和图片
       - 问答内容重复或质量较低
       - 机构介绍、地点查询等非技术性内容
    
    特别注意:
    - 统计方法、数学模型、算法理论等内容一律保留
    - 参数估计、损失函数、信息矩阵等技术概念必须保留
    - 不要因为内容专业或包含数学符号而删除

    输出格式:
    仅返回主题描述文本

    主题描述:"""

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个擅长总结概括的助手"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=30
            )
            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"第{attempt + 1}次尝试失败: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避
                print(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数，使用默认instruction")
                return "解释这个概念"


def convert_to_alpaca_format(input_file: str, output_file: str, api_key: str) -> None:
    """
    将自定义JSON格式转换为Alpaca格式

    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        api_key: 阿里云百炼API密钥
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 临时文件路径
    temp_file = output_file + ".temp"

    # 尝试加载未完成的处理进度
    start_index = 0
    alpaca_data = []
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                alpaca_data = json.load(f)
            start_index = len(alpaca_data)
            print(f"检测到未完成的处理，从第 {start_index + 1} 条继续...")
        except Exception as e:
            print(f"加载临时文件失败: {str(e)}，将从头开始处理")

    # 记录开始时间
    start_time = time.time()
    last_save_time = start_time

    for idx in range(start_index, len(data)):
        item = data[idx]
        input_text = item["question"]
        output_text = item["answer"]

        # 显示进度（每100条显示一次）
        if (idx + 1) % 100 == 0 or (idx + 1) == len(data):
            elapsed = time.time() - start_time
            print(f"处理进度: {idx + 1}/{len(data)} ({((idx + 1) / len(data)) * 100:.1f}%) 已用时: {elapsed:.1f}秒")

        instruction = generate_instruction_with_qwen(input_text, output_text, api_key)
        # 清理可能包含的前缀
        if instruction.startswith("主题描述："):
            instruction = instruction.replace("主题描述：", "").strip()
        if instruction.startswith("**主题描述**："):
            instruction = instruction.replace("**主题描述**：", "").strip()
        if instruction.startswith("instruction："):
            instruction = instruction.replace("instruction：", "").strip()

        # 移除任何评估内容（按换行分割取第一部分）
        if "\n" in instruction:
            instruction = instruction.split("\n")[0].strip()

        if instruction.strip().upper() == "DELETE":
            print(f"跳过第 {idx + 1} 条数据：{input_text[:50]}...")
            continue  # 跳过完全无关的数据

        alpaca_item = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }

        alpaca_data.append(alpaca_item)

        # 每1000条或最后一条保存一次
        if (idx + 1) % 100 == 0 or (idx + 1) == len(data):
            current_time = time.time()
            if current_time - last_save_time > 30:  # 至少30秒保存一次
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
                print(f"已保存处理到第 {idx + 1} 条到临时文件")
                last_save_time = current_time

    # 处理完成后重命名临时文件
    os.rename(temp_file, output_file)
    print(f"所有数据处理完成，最终结果已保存到 {output_file}")


if __name__ == "__main__":
    # 配置参数
    input_json = "datasets-literature-label-2025-09-12.json"
    output_json = "alpaca_format_qwen.json"

    # 获取API密钥（优先从环境变量获取）
    dashscope_api_key = "sk-4bc62e82d5084ce78b557bdea04aaa21"


    # 执行转换
    print("开始转换数据...")
    try:
        convert_to_alpaca_format(input_json, output_json, dashscope_api_key)
    except KeyboardInterrupt:
        print("\n用户中断处理，已保存当前进度到临时文件")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        print("当前进度已保存到临时文件，可以重新运行程序继续处理")


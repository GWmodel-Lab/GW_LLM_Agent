import json
import time
import os
import glob
from typing import Dict, List, Any
from openai import OpenAI


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

def process_alpaca_format_with_qwen(data_item: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """使用Qwen模型处理单条alpaca格式数据，生成格式转换结果"""
    
    instruction = data_item.get("instruction", "")
    input_text = data_item.get("input", "")
    output = data_item.get("output", "")
    
    # 如果input为空，使用instruction作为问题
    question = input_text if input_text.strip() else instruction
    answer = output
    
    # 生成新的instruction
    new_instruction = generate_instruction_with_qwen(question, answer, api_key)
    
    # 清理可能包含的前缀
    if new_instruction.startswith("主题描述："):
        new_instruction = new_instruction.replace("主题描述：", "").strip()
    if new_instruction.startswith("**主题描述**："):
        new_instruction = new_instruction.replace("**主题描述**：", "").strip()
    if new_instruction.startswith("instruction："):
        new_instruction = new_instruction.replace("instruction：", "").strip()

    # 移除任何评估内容（按换行分割取第一部分）
    if "\n" in new_instruction:
        new_instruction = new_instruction.split("\n")[0].strip()

    # 如果返回DELETE标记，跳过这条数据
    if new_instruction.strip().upper() == "DELETE":
        return None

    # 返回重新格式化的数据
    return {
        "instruction": new_instruction,
        "input": question,
        "output": answer
    }

def convert_to_alpaca_format(input_file: str, output_file: str, api_key: str) -> None:
    """
    将现有alpaca格式数据重新优化为更好的alpaca格式

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

        # 显示进度（每10条显示一次）
        if (idx + 1) % 10 == 0 or (idx + 1) == len(data):
            elapsed = time.time() - start_time
            print(f"处理进度: {idx + 1}/{len(data)} ({((idx + 1) / len(data)) * 100:.1f}%) 已用时: {elapsed:.1f}秒")

        # 使用Qwen模型处理数据
        processed_item = process_alpaca_format_with_qwen(item, api_key)
        
        if processed_item is None:
            print(f"跳过第 {idx + 1} 条数据：{item.get('instruction', '')[:50]}...")
            continue  # 跳过被标记为DELETE的数据

        alpaca_data.append(processed_item)

        # 每50条或最后一条保存一次
        if (idx + 1) % 50 == 0 or (idx + 1) == len(data):
            current_time = time.time()
            if current_time - last_save_time > 30:  # 至少30秒保存一次
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
                print(f"已保存处理到第 {idx + 1} 条到临时文件")
                last_save_time = current_time

    # 处理完成后重命名临时文件
    if os.path.exists(temp_file):
        os.rename(temp_file, output_file)
    print(f"所有数据处理完成，最终结果已保存到 {output_file}")


def batch_convert_json_files(base_folder: str, result_folder: str, api_key: str) -> None:
    """
    批量转换base文件夹中的所有.json文件

    参数:
        base_folder: 输入文件夹路径
        result_folder: 输出文件夹路径
        api_key: 阿里云百炼API密钥
    """
    # 确保输出文件夹存在
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"创建输出文件夹: {result_folder}")

    # 查找base文件夹中的所有.json文件
    json_pattern = os.path.join(base_folder, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"在 {base_folder} 文件夹中未找到任何.json文件")
        return

    print(f"找到 {len(json_files)} 个.json文件需要处理:")
    for file_path in json_files:
        print(f"  - {os.path.basename(file_path)}")

    # 处理每个文件
    for i, input_file in enumerate(json_files, 1):
        print(f"\n{'='*60}")
        print(f"正在处理第 {i}/{len(json_files)} 个文件: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(result_folder, f"new-{base_name}.json")
        
        try:
            convert_to_alpaca_format(input_file, output_file, api_key)
            print(f"✅ 文件 {os.path.basename(input_file)} 处理完成，结果保存到 {os.path.basename(output_file)}")
        except Exception as e:
            print(f"❌ 处理文件 {os.path.basename(input_file)} 时发生错误: {str(e)}")
            continue

    print(f"\n🎉 批量转换完成！所有结果已保存到 {result_folder} 文件夹")

def main():
    """主函数"""
    # 配置参数
    base_folder = "base"
    result_folder = "result"

    # 获取API密钥
    dashscope_api_key = "sk-0d8bb60c1dea41708e5634926d570b5d"

    # 检查base文件夹是否存在
    if not os.path.exists(base_folder):
        print(f"错误：输入文件夹 {base_folder} 不存在")
        print("请确保在base文件夹中放置需要转换的.json文件")
        return

    # 执行批量转换
    print("开始批量转换数据...")
    try:
        batch_convert_json_files(base_folder, result_folder, dashscope_api_key)
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"批量处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()

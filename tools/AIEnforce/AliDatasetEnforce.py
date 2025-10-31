import os
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI

# 四种语境配置
CONTEXTS: List[Dict[str, str]] = [
    {
        "id": "classroom_qa",
        "user": "学生",
        "assistant": "老师",
        "scenario": "课堂学生提问问答"
    },
    {
        "id": "academic_discussion",
        "user": "地理信息领域学者",
        "assistant": "学者",
        "scenario": "学术问题讨论"
    },
    {
        "id": "engineer_peer",
        "user": "工程师",
        "assistant": "工程师",
        "scenario": "项目进度讨论/项目方法讨论"
    },
    {
        "id": "engineer_client",
        "user": "工程师",
        "assistant": "客户",
        "scenario": "工程项目内容汇报/工程项目方法解释"
    },
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or os.getenv("DASHSCOPE_API_KEY") or ""
    if not key:
        # 兼容：若未设置环境变量，尝试使用脚本内置密钥（如需）
        key = "sk-0d8bb60c1dea41708e5634926d570b5d"
    return OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def build_prompt(context: Dict[str, str], item: Dict[str, Any]) -> str:
    instruction = item.get("instruction", "").strip()
    input_text = item.get("input", "").strip()
    output_text = item.get("output", "").strip()

    # 若input为空，则将instruction视为提问内容语义依据
    question_basis = input_text if input_text else instruction

    prompt = f"""
你是{context['assistant']}，对话场景：{context['scenario']}。

数据集为地理信息、地理加权回归与空间分析领域的微调样本（alpaca格式）。
在不修改原有问答内容、不偏离原instruction语义的前提下，为该样本在本场景下扩充一段两轮（共4条消息）的history对话。

约束与要求：
1) 严禁篡改原问答：原instruction/input/output必须保持语义一致且不被改写。
2) 对话身份：history中轮流出现用户（{context['user']}）与助理（{context['assistant']}），共2轮=4条消息，从用户开场。
3) 语气与内容：严格契合场景“{context['scenario']}”，围绕原问题主题展开追问、澄清、示例与解释，逐步深入。
4) 一致性：回答口吻、术语与领域一致，不跳出地理信息/GWR/空间分析范畴。
5) 若本场景下无法合理扩充（比如与角色或场景强冲突），请返回 JSON {{"skip": true}}。
6) 输出必须是严格JSON，字段：
   {{
     "skip": false|true,
     "system": "{context['assistant']}",
     "histories": [
       {{"role": "user", "content": "..."}},
       {{"role": "assistant", "content": "..."}},
       ... 共4条
     ]
   }}

原样本：
- instruction: {instruction}
- input: {question_basis}
- output: {output_text}

注意：
- histories应与上述原样本主题一致，可围绕input与output展开上下文追问；
- histories不得直接复制原output全文，可引用其中观点进行解释或层次化说明；
- 严格输出JSON，勿添加解释性文字或Markdown标记。
""".strip()
    return prompt


def call_api_for_context(client: OpenAI, context: Dict[str, str], item: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
    prompt = build_prompt(context, item)

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": f"你现在扮演{context['assistant']}，请严格遵循指令并只输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1200,
            )
            content = completion.choices[0].message.content.strip()
            # 解析JSON
            data = json.loads(content)
            # 基本校验
            if data.get("skip") is True:
                return None
            if data.get("system") != context["assistant"]:
                return None
            histories = data.get("histories", []) or data.get("history", [])
            if not isinstance(histories, list) or len(histories) != 4:
                return None
            # 验证role交替
            roles_ok = all(h.get("role") in ("user", "assistant") for h in histories)
            if not roles_ok:
                return None
            # 简单交替检查：奇数位user，偶数位assistant
            for idx, h in enumerate(histories):
                expect = "user" if idx % 2 == 0 else "assistant"
                if h.get("role") != expect:
                    return None
            return data
        except Exception as e:
            wait = (attempt + 1) * 2
            print(f"调用失败(第{attempt + 1}次)：{e}，{wait}s后重试...")
            time.sleep(wait)
    return None


def process_dataset(input_path: str, output_path: str, api_key: Optional[str] = None) -> None:
    client = create_client(api_key)

    with open(input_path, 'r', encoding='utf-8') as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []
    temp_path = output_path + ".temp"

    # 若存在临时文件，尝试断点续跑
    start_index = 0
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r', encoding='utf-8') as tf:
                exist = json.load(tf)
            if isinstance(exist, list):
                results = exist
                start_index = len(results)
                print(f"检测到未完成进度，从第{start_index + 1}条继续...")
        except Exception:
            pass

    start_time = time.time()
    last_save = start_time

    for idx in range(start_index, len(dataset)):
        item = dataset[idx]
        print(f"处理 {idx + 1}/{len(dataset)}: {item.get('instruction', '')[:40]}...")

        expanded_rows: List[Dict[str, Any]] = []
        for ctx in CONTEXTS:
            data = call_api_for_context(client, ctx, item)
            if data is not None:
                histories = data.get("histories", []) or data.get("history", [])
                expanded_rows.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "output": item.get("output", ""),
                    "system": data.get("system"),
                    "history": histories
                })
            else:
                print(f"  - 跳过场景: {ctx['id']}")

        # 将每个语境的独立记录追加到总结果
        results.extend(expanded_rows)

        # 定期保存临时文件
        if (idx + 1) % 10 == 0 or (time.time() - last_save) > 30:
            with open(temp_path, 'w', encoding='utf-8') as tf:
                json.dump(results, tf, ensure_ascii=False, indent=2)
            last_save = time.time()
            elapsed = int(last_save - start_time)
            print(f"已处理 {idx + 1}/{len(dataset)}（展开后累计{len(results)}条），用时 {elapsed}s，临时进度已保存。")

    # 写出最终文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"完成！结果写入 {output_path}")


if __name__ == "__main__":
    input_file = "test.json"
    output_file = "testEnforce.json"
    # 可通过环境变量DASHSCOPE_API_KEY设置API Key
    process_dataset(input_file, output_file)

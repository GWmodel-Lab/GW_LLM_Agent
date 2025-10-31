import os
import json
import time
import asyncio
import aiohttp
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@dataclass
class APIResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AsyncQwenClient:
    """异步Qwen API客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1", semaphore_size: int = 5):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(semaphore_size)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def build_prompt(self, context: Dict[str, str], item: Dict[str, Any]) -> str:
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()
        question_basis = input_text if input_text else instruction

        prompt = f"""
你是{context['assistant']}，对话场景：{context['scenario']}。

数据集为地理信息、地理加权回归与空间分析领域的微调样本（alpaca格式）。
在不修改原有问答内容、不偏离原instruction语义的前提下，为该样本在本场景下扩充一段两轮（共4条消息）的history对话。

约束与要求：
1) 严禁篡改原问答：原instruction/input/output必须保持语义一致且不被改写。
2) 对话身份：history中轮流出现用户（{context['user']}）与助理（{context['assistant']}），共2轮=4条消息，从用户开场。
3) 语气与内容：严格契合场景"{context['scenario']}"，围绕原问题主题展开追问、澄清、示例与解释，逐步深入。
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

    async def call_api_single(self, context: Dict[str, str], item: Dict[str, Any], max_retries: int = 3) -> APIResponse:
        async with self.semaphore:
            prompt = self.build_prompt(context, item)
            for attempt in range(max_retries):
                try:
                    payload = {
                        "model": "qwen-plus",
                        "messages": [
                            {"role": "system", "content": f"你现在扮演{context['assistant']}，请严格遵循指令并只输出JSON。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 1200,
                    }
                    async with self.session.post(f"{self.base_url}/chat/completions", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content'].strip()
                            data = json.loads(content)
                            if data.get("skip") is True:
                                return APIResponse(success=False, error="场景跳过")
                            if data.get("system") != context["assistant"]:
                                return APIResponse(success=False, error="system不匹配")
                            histories = data.get("histories", []) or data.get("history", [])
                            if not isinstance(histories, list) or len(histories) != 4:
                                return APIResponse(success=False, error="histories长度不正确")
                            roles_ok = all(h.get("role") in ("user", "assistant") for h in histories)
                            if not roles_ok:
                                return APIResponse(success=False, error="role不正确")
                            for idx, h in enumerate(histories):
                                expect = "user" if idx % 2 == 0 else "assistant"
                                if h.get("role") != expect:
                                    return APIResponse(success=False, error="role交替错误")
                            return APIResponse(success=True, data=data)
                        else:
                            errt = await response.text()
                            logger.warning(f"API状态{response.status}: {errt}")
                except Exception as e:
                    logger.warning(f"API异常(第{attempt + 1}次)：{e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 2)
                    else:
                        return APIResponse(success=False, error=str(e))
            return APIResponse(success=False, error="达到最大重试次数")


async def process_single_item(client: AsyncQwenClient, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    expanded_rows: List[Dict[str, Any]] = []
    tasks = [client.call_api_single(ctx, item) for ctx in CONTEXTS]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for i, resp in enumerate(responses):
        ctx = CONTEXTS[i]
        if isinstance(resp, Exception):
            logger.warning(f"语境 {ctx['id']} 异常: {resp}")
            continue
        if resp.success and resp.data:
            histories = resp.data.get("histories", []) or resp.data.get("history", [])
            expanded_rows.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "system": resp.data.get("system"),
                "history": histories
            })
        else:
            logger.info(f"跳过语境 {ctx['id']}: {resp.error}")
    return expanded_rows

async def process_items_async(items: List[Dict[str, Any]], api_key: str, batch_size: int = 8, semaphore_size: int = 5) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    async with AsyncQwenClient(api_key, semaphore_size=semaphore_size) as client:
        for start in range(0, len(items), batch_size):
            end = min(start + batch_size, len(items))
            batch = items[start:end]
            batch_tasks = [process_single_item(client, it) for it in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for br in batch_results:
                if isinstance(br, Exception):
                    logger.warning(f"子批异常: {br}")
                    continue
                results.extend(br)
    return results


# -------------- 多进程封装 --------------
def _split_chunks(data: List[Dict[str, Any]], parts: int) -> List[List[Dict[str, Any]]]:
    if parts <= 1:
        return [data]
    n = len(data)
    base = n // parts
    rem = n % parts
    chunks: List[List[Dict[str, Any]]] = []
    s = 0
    for i in range(parts):
        e = s + base + (1 if i < rem else 0)
        if s < e:
            chunks.append(data[s:e])
        s = e
    return chunks


def _worker_entry(args):
    items, api_key, batch_size, sem_size, wid, total = args
    logger.info(f"[Worker {wid}/{total}] 启动，分片: {len(items)} 条")
    out = asyncio.run(process_items_async(items, api_key, batch_size=batch_size, semaphore_size=sem_size))
    logger.info(f"[Worker {wid}/{total}] 完成，产出: {len(out)} 条")
    return out


def process_file(input_file: str, output_file: str, workers: Optional[int] = None, per_batch: int = 6, sem_size: int = 5) -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-0d8bb60c1dea41708e5634926d570b5d"
    if not api_key:
        logger.error("未找到API密钥，请设置环境变量DASHSCOPE_API_KEY或修改脚本中的默认密钥")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    cpu_cnt = os.cpu_count() or mp.cpu_count() or 1
    auto_workers = max(1, min(cpu_cnt, len(dataset)))
    use_workers = max(1, min(workers if workers else auto_workers, len(dataset)))
    logger.info(f"文件: {os.path.basename(input_file)} | CPU: {cpu_cnt} | 进程数: {use_workers}")

    chunks = _split_chunks(dataset, use_workers)
    args_list = [(chunks[i], api_key, per_batch, sem_size, i + 1, use_workers) for i in range(len(chunks))]

    start = time.time()
    results: List[Dict[str, Any]] = []
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with mp.get_context("spawn").Pool(processes=use_workers) as pool:
        for part in pool.imap_unordered(_worker_entry, args_list):
            results.extend(part)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = int(time.time() - start)
    logger.info(f"完成 | 输出: {output_file} | 用时 {elapsed}s | 记录 {len(results)}")


def main():
    input_file = "DatasetCorrected/new-R package-training.json"
    output_file = "DatasetEnforce/en-R package-training-test.json"
    process_file(input_file, output_file)


if __name__ == "__main__":
    main()

import os
import json
import time
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# 配置日志
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
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(5)  # 限制并发数
        
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
        """构建提示词"""
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
        """单个API调用"""
        async with self.semaphore:  # 限制并发
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
                            
                            # 解析JSON
                            data = json.loads(content)
                            
                            # 基本校验
                            if data.get("skip") is True:
                                return APIResponse(success=False, error="场景跳过")
                            
                            if data.get("system") != context["assistant"]:
                                return APIResponse(success=False, error="system字段不匹配")
                            
                            histories = data.get("histories", []) or data.get("history", [])
                            if not isinstance(histories, list) or len(histories) != 4:
                                return APIResponse(success=False, error="histories长度不正确")
                            
                            # 验证role交替
                            roles_ok = all(h.get("role") in ("user", "assistant") for h in histories)
                            if not roles_ok:
                                return APIResponse(success=False, error="role字段不正确")
                            
                            # 简单交替检查：奇数位user，偶数位assistant
                            for idx, h in enumerate(histories):
                                expect = "user" if idx % 2 == 0 else "assistant"
                                if h.get("role") != expect:
                                    return APIResponse(success=False, error="role交替不正确")
                            
                            return APIResponse(success=True, data=data)
                        else:
                            error_text = await response.text()
                            logger.warning(f"API调用失败，状态码: {response.status}, 错误: {error_text}")
                            
                except Exception as e:
                    logger.warning(f"API调用异常(第{attempt + 1}次)：{e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        await asyncio.sleep(wait_time)
                    else:
                        return APIResponse(success=False, error=str(e))
            
            return APIResponse(success=False, error="达到最大重试次数")
    
    async def call_api_batch(self, context: Dict[str, str], items: List[Dict[str, Any]]) -> List[APIResponse]:
        """批量API调用"""
        tasks = [self.call_api_single(context, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)


async def process_single_item(client: AsyncQwenClient, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """处理单个数据项的所有语境"""
    expanded_rows = []
    
    # 为所有语境创建并发任务
    tasks = [client.call_api_single(context, item) for context in CONTEXTS]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, response in enumerate(responses):
        context = CONTEXTS[i]
        
        if isinstance(response, Exception):
            logger.warning(f"语境 {context['id']} 处理异常: {response}")
            continue
            
        if response.success and response.data:
            histories = response.data.get("histories", []) or response.data.get("history", [])
            expanded_rows.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "system": response.data.get("system"),
                "history": histories
            })
        else:
            logger.info(f"跳过语境 {context['id']}: {response.error}")
    
    return expanded_rows


async def process_dataset_async(input_path: str, output_path: str, api_key: str, batch_size: int = 10) -> None:
    """异步处理数据集"""
    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset: List[Dict[str, Any]] = json.load(f)
    
    logger.info(f"开始处理 {len(dataset)} 条数据，批次大小: {batch_size}")
    
    results: List[Dict[str, Any]] = []
    temp_path = output_path + ".temp"
    
    # 检查断点续跑
    start_index = 0
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r', encoding='utf-8') as tf:
                existing_results = json.load(tf)
            if isinstance(existing_results, list):
                results = existing_results
                start_index = len(results) // 4  # 每个原始项展开为4个语境
                logger.info(f"检测到未完成进度，从第 {start_index + 1} 条继续...")
        except Exception as e:
            logger.warning(f"加载临时文件失败: {e}")
    
    start_time = time.time()
    last_save = start_time
    
    async with AsyncQwenClient(api_key) as client:
        # 分批处理
        for batch_start in range(start_index, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_items = dataset[batch_start:batch_end]
            
            logger.info(f"处理批次 {batch_start + 1}-{batch_end}/{len(dataset)}")
            
            # 并发处理批次内的所有项目
            batch_tasks = [process_single_item(client, item) for item in batch_items]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 收集结果
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"处理第 {batch_start + i + 1} 条数据时发生异常: {result}")
                    continue
                results.extend(result)
            
            # 定期保存
            current_time = time.time()
            if (batch_end % (batch_size * 2) == 0) or (current_time - last_save > 60):
                with open(temp_path, 'w', encoding='utf-8') as tf:
                    json.dump(results, tf, ensure_ascii=False, indent=2)
                last_save = current_time
                elapsed = int(current_time - start_time)
                logger.info(f"已处理 {batch_end}/{len(dataset)}（展开后累计{len(results)}条），用时 {elapsed}s，临时进度已保存。")
    
    # 保存最终结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    total_time = int(time.time() - start_time)
    logger.info(f"完成！结果写入 {output_path}，总用时 {total_time}s，共生成 {len(results)} 条记录")


async def main():
    """主函数"""
    input_file = "DatasetCorrected/new-R package-training.json"
    output_file = "DatasetEnforce/en-R package-training.json"
    
    # 获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY") or "sk-0d8bb60c1dea41708e5634926d570b5d"
    
    if not api_key:
        logger.error("未找到API密钥，请设置环境变量DASHSCOPE_API_KEY或修改脚本中的默认密钥")
        return
    
    try:
        await process_dataset_async(input_file, output_file, api_key, batch_size=5)
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())

import json
import os
import sys


def to_messages(item: dict):
    messages = []
    system = item.get("system")
    instr = item.get("instruction", "")
    inp = item.get("input", "")
    out = item.get("output", "")

    if system and isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    if inp and isinstance(inp, str) and inp.strip():
        user_content = f"指令:\n{instr}\n\n输入:\n{inp}"
    else:
        user_content = instr

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": out})
    return messages


def convert_file(src_path: str):
    dir_name = os.path.dirname(src_path)
    base = os.path.basename(src_path)
    dst_path = os.path.join(dir_name, f"new_{base}")

    with open(src_path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 容错：将按顺序提取顶层对象，视为JSONL或逗号缺失的对象序列
        objs = []
        buf = []
        depth = 0
        in_str = False
        esc = False
        for ch in raw:
            buf.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        # 尝试解析当前对象
                        candidate = ''.join(buf).strip()
                        try:
                            obj = json.loads(candidate)
                            objs.append(obj)
                        except Exception:
                            pass
                        buf = []
        data = objs

    # 输入可能是数组或对象（容错）
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("不支持的JSON顶层结构，应为数组或包含data的对象。")

    out_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # 统一键名的轻量容错
        norm = {
            "instruction": it.get("instruction", it.get("prompt", "")),
            "input": it.get("input", it.get("context", "")),
            "output": it.get("output", it.get("response", "")),
            "system": it.get("system", it.get("system_prompt", "")),
        }
        msgs = to_messages(norm)
        out_items.append({"messages": msgs})

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)

    return dst_path, len(items), len(out_items)


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/convert_chatml.py <file1> [file2 ...]")
        sys.exit(1)

    for p in sys.argv[1:]:
        dst, n_in, n_out = convert_file(p)
        print(f"转换完成: {p} -> {dst} | 条目: {n_in} -> {n_out}")


if __name__ == "__main__":
    main()



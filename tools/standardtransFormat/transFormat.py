import json
import os
import sys


def old_to_new_item(item: dict) -> dict:
    system = item.get("system", item.get("system_prompt", ""))
    instr = item.get("instruction", item.get("prompt", ""))
    inp = item.get("input", item.get("context", ""))
    out = item.get("output", item.get("response", ""))

    messages = []
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})

    if isinstance(inp, str) and inp.strip():
        user_content = f"指令:\n{instr}\n\n输入:\n{inp}"
    else:
        user_content = instr
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": out})

    return {"messages": messages}


def new_to_old_item(sample: dict) -> dict:
    messages = sample.get("messages") or []
    system = ""
    instruction = ""
    inp = ""
    output = ""

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system" and not system:
            system = content
        elif role == "user" and not instruction:
            # 反向拆分“指令/输入”
            if content.startswith("指令:\n"):
                body = content[len("指令:\n"):]
                if "\n\n输入:\n" in body:
                    instruction, inp = body.split("\n\n输入:\n", 1)
                else:
                    instruction = body
            else:
                instruction = content
        elif role == "assistant" and not output:
            output = content

    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "system": system
    }


def detect_format(obj):
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # new: [{messages:[...]}, ...]
        if "messages" in obj[0]:
            return "new"
        # old: [{instruction:..., input:..., output:...}, ...]
        if "instruction" in obj[0] and "output" in obj[0]:
            return "old"
    return None


def load_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 容错：按对象片段解析
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
                        candidate = ''.join(buf).strip()
                        try:
                            obj = json.loads(candidate)
                            objs.append(obj)
                        except Exception:
                            pass
                        buf = []
        return objs


def convert(in_path: str, direction: str, out_path: str = None):
    data = load_json_any(in_path)
    fmt = detect_format(data)
    if direction == "auto":
        direction = "old2new" if fmt == "old" else ("new2old" if fmt == "new" else None)
        if direction is None:
            raise ValueError("无法自动判断数据格式，请指定 --dir old2new 或 new2old")

    if direction == "old2new":
        out = [old_to_new_item(x) for x in data]
    elif direction == "new2old":
        out = [new_to_old_item(x) for x in data]
    else:
        raise ValueError("direction 只能是 old2new/new2old/auto")

    if not out_path:
        d = os.path.dirname(in_path)
        b = os.path.basename(in_path)
        out_path = os.path.join(d, b)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out_path, len(data), len(out)


def batch_convert(src_dir: str, dst_dir: str, target: str = "new", dir_mode: str = "auto"):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for name in os.listdir(src_dir):
        if not name.lower().endswith('.json'):
            continue
        in_path = os.path.join(src_dir, name)
        if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
            continue
        data = load_json_any(in_path)
        fmt = detect_format(data)
        if target == "new":
            direction = "old2new" if fmt == "old" else ("new2old" if fmt == "new" and dir_mode == "new2old" else None)
            if direction is None and fmt == "new":
                out_path = os.path.join(dst_dir, name)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"已是新格式: {in_path} -> {out_path}")
                count += 1
                continue
        elif target == "old":
            direction = "new2old" if fmt == "new" else ("old2new" if fmt == "old" and dir_mode == "old2new" else None)
            if direction is None and fmt == "old":
                out_path = os.path.join(dst_dir, name)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"已是旧格式: {in_path} -> {out_path}")
                count += 1
                continue
        else:
            if dir_mode == "auto":
                direction = "old2new" if fmt == "old" else ("new2old" if fmt == "new" else None)
            else:
                direction = dir_mode
        out_path = os.path.join(dst_dir, name)
        dst, n_in, n_out = convert(in_path, direction, out_path)
        print(f"转换完成: {in_path} -> {dst} | 条目: {n_in} -> {n_out}")
        count += 1
    print(f"批量完成，共处理JSON文件: {count}")


def main():
    # 固定的目录到目录转换，不使用argparse，不提供单文件模式
    SRC_DIR = "D:/project/GW_LLM_Agent-main/对话式数据集"
    DST_DIR = "D:/project/GW_LLM_Agent-main/对话式数据集新"
    TARGET = "new"    # 固定目标为新格式
    DIR_MODE = "auto" # 自动识别旧/新
    batch_convert(SRC_DIR, DST_DIR, TARGET, DIR_MODE)


if __name__ == "__main__":
    main()



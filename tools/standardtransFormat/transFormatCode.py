# -- coding: utf-8 --

import json
import os


def _read_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _extract_sections_zh(text: str) -> dict:
    """
    从文档中提取 1~6 节的中文内容块。
    返回字典：{
      1: "概述-中文",
      2: "类成员-中文",
      3: "主要方法-中文",
      4: "算法流程-中文",
      5: "特点-中文",
      6: "应用场景-中文"
    }
    """
    lines = text.splitlines()

    # 支持两种格式：
    # A) 带数字标题 + “中文/English” 标记
    # B) 方括号标题 [概述]/[类成员]/[主要方法]/[算法流程]/[特点]/[应用场景]，区块内中英文混排

    def has_chinese(s: str) -> bool:
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)

    # 先尝试格式 A
    anchors_a = {
        1: '1. 概述',
        2: '2. 类成员',
        3: '3. 主要方法',
        4: '4. 算法流程',
        5: '5. 特点',
        6: '6. 应用场景',
    }
    starts_a = {}
    for idx, line in enumerate(lines):
        for k, kw in anchors_a.items():
            if line.strip().startswith(kw):
                starts_a[k] = idx
    if len(starts_a) >= 3:  # 认为是格式 A
        ends_a = {}
        sorted_keys = sorted(starts_a.keys())
        for i, k in enumerate(sorted_keys):
            if i + 1 < len(sorted_keys):
                ends_a[k] = starts_a[sorted_keys[i + 1]]
            else:
                ends_a[k] = len(lines)

        def extract_cn_block_a(block_lines):
            cn_start = None
            for i, l in enumerate(block_lines):
                if l.strip() == '中文':
                    cn_start = i + 1
                    break
            if cn_start is None:
                # 兜底：若缺少“中文”，提取到"English"之前且包含中文的行
                buf = []
                for l in block_lines:
                    if l.strip() == 'English':
                        break
                    if has_chinese(l):
                        buf.append(l)
                return '\n'.join(buf).strip()
            cn_end = len(block_lines)
            for j in range(cn_start, len(block_lines)):
                if block_lines[j].strip() == 'English':
                    cn_end = j
                    break
            content = '\n'.join(block_lines[cn_start:cn_end]).strip('\n').strip()
            return content

        result = {}
        for k in sorted_keys:
            block = lines[starts_a[k]:ends_a[k]]
            result[k] = extract_cn_block_a(block)
        return result

    # 否则，尝试格式 B：方括号标题
    title_map_b = {
        '概述': 1,
        '类成员': 2,
        '主要方法': 3,
        '算法流程': 4,
        '特点': 5,
        '应用场景': 6,
    }
    # 收集每个标题的起始行
    starts_b = []  # list of (section_id, index)
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            key = s[1:-1].strip()
            if key in title_map_b:
                starts_b.append((title_map_b[key], idx))
    if not starts_b:
        return {}
    starts_b.sort(key=lambda x: x[1])
    # 计算结束行
    ranges = []  # (section_id, start, end)
    for i, (sec_id, st) in enumerate(starts_b):
        if i + 1 < len(starts_b):
            en = starts_b[i + 1][1]
        else:
            en = len(lines)
        ranges.append((sec_id, st + 1, en))  # 内容从标题下一行开始

    result = {1: '', 2: '', 3: '', 4: '', 5: '', 6: ''}
    for sec_id, st, en in ranges:
        block = lines[st:en]
        # 仅收集含中文的行，自动忽略英文
        buf = [l for l in block if has_chinese(l)]
        # 去掉首尾空行
        content = '\n'.join([l for l in buf]).strip()
        result[sec_id] = content
    return result


def _infer_class_name_from_path(doc_path: str) -> str:
    base = os.path.splitext(os.path.basename(doc_path))[0]
    # 约定：去掉尾部的 "_Documentation" 后缀得到类名
    suffix = '_Documentation'
    if base.endswith(suffix):
        return base[: -len(suffix)]
    return base


def build_sample_from_doc(doc_path: str) -> dict:
    text = _read_text(doc_path)
    sec = _extract_sections_zh(text)
    class_name = _infer_class_name_from_path(doc_path)

    overview = sec.get(1, '')
    members = sec.get(2, '')
    methods = sec.get(3, '')
    workflow = sec.get(4, '')
    features = sec.get(5, '')
    scenarios = sec.get(6, '')

    system = f'你是一位空间数据分析领域的学术研究者。当前对象：{class_name} 类。'
    instruction = f'结合文档给出的 {class_name} 类的实际应用'
    inp = f'请基于 {class_name} 类说明文档，总结其用途、成员、方法、流程、特点与应用场景，并给出伪代码示例。'

    # 产出：中文总结 + 简短伪代码使用示例
    output_parts = []
    if overview:
        output_parts.append('【概述】\n' + overview)
    if members:
        output_parts.append('【类成员】\n' + members)
    # 注意：示例 BandwidthSelector_formatted.json 中，output 不包含【关键方法】
    # 因此此处不将 methods 拼入 output，仅保留在 history 中
    if workflow:
        output_parts.append('【算法流程】\n' + workflow)
    if features:
        output_parts.append('【特点】\n' + features)
    if scenarios:
        output_parts.append('【应用场景】\n' + scenarios)

    pseudo_code = (
        '【伪代码示例】\n'
        '1) 初始化带宽上下界与核函数\n'
        '2) 调用 optimize(instance) 迭代搜索最优带宽\n'
        '3) 读取 bandwidthCriterion() 以绘制 带宽-准则曲线\n'
    )
    output_parts.append(pseudo_code)
    output = '\n\n'.join([p for p in output_parts if p])

    history = [
        [f'第一轮指令（什么是{class_name}）', overview or ''],
        [f'第二轮指令（{class_name}有哪些类成员）', members or ''],
        [f'第三轮指令（{class_name}有哪些关键方法）', methods or ''],
        [f'第四轮指令（{class_name}的算法流程是什么）', workflow or ''],
        [f'第五轮指令（{class_name}具有什么特点）', features or ''],
        [f'第六轮指令（{class_name}的应用场景有哪些）', scenarios or ''],
    ]

    sample = {
        'instruction': instruction,
        'input': inp,
        'output': output,
        'system': system,
        'history': history,
    }
    return sample


def save_dataset(sample: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump([sample], f, ensure_ascii=False, indent=2)


def main():
    tools_dir = os.path.dirname(__file__)
    src_dir = os.path.join(tools_dir, 'GWModel_R_Documentation')
    dst_dir = os.path.join(tools_dir, 'codeDataset')
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历源目录中的所有 txt 文档
    names = [n for n in os.listdir(src_dir) if n.lower().endswith('.txt')]
    total = 0
    for name in names:
        in_path = os.path.join(src_dir, name)
        base = os.path.splitext(name)[0]
        out_name = f'{base}_formatted.json'
        out_path = os.path.join(dst_dir, out_name)

        sample = build_sample_from_doc(in_path)
        save_dataset(sample, out_path)
        print(f'已生成: {out_path}')
        total += 1

    print(f'批量完成，共处理: {total} 个文档')


if __name__ == '__main__':
    main()



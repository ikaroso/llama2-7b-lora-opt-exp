"""Validate and summarize five paired rounds of the fixed-length Spec-Bench subset."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


METHODS = ('reference', 'ngram', 'draft3')
CATEGORIES = ('summarization', 'rag')
SCOPES = ('all',) + CATEGORIES
METRICS = ('prefill_tps', 'decode_tps', 'ttft_ms', 'wall_seconds')
ROUNDS = range(1, 6)
SAMPLE_COUNT = 12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f'Missing evidence: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    return dict(values=values, mean=statistics.mean(values),
                stdev=statistics.stdev(values), min=min(values), max=max(values))


def positive(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0, f'Invalid {name}: {value!r}')
    return value


def count(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f'Invalid {name}: {value!r}')
    return value


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def first_difference(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def returned_text(row, name):
    events = row.get('events')
    require(isinstance(events, list) and events, f'{name}: no SSE evidence')
    require(events[-1]['data'].get('stop') is True, f'{name}: incomplete SSE stream')
    require(events[-1]['data'] == row['final'], f'{name}: final/SSE mismatch')
    parts = [event['data'].get('content', '') for event in events]
    require(all(isinstance(part, str) for part in parts), f'{name}: invalid SSE content')
    # Compare the actual returned text, without retokenizing or treating partial
    # streaming token arrays as complete generated token sequences.
    return ''.join(parts)


def telemetry(path):
    result = dict(available=False, samples=0, whole_gpu_memory_max_mib=None,
                  temperature_max_c=None, power_max_w=None, sm_clock_max_mhz=None)
    if not path.is_file():
        return result
    columns = [[], [], [], []]
    with path.open(encoding='utf-8') as source:
        for row in csv.reader(source):
            if len(row) != 5:
                continue
            parsed = False
            for values, text in zip(columns, row[1:]):
                try:
                    value = float(text.strip())
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
                    parsed = True
            result['samples'] += int(parsed)
    for key, values in zip(('whole_gpu_memory_max_mib', 'temperature_max_c',
                            'power_max_w', 'sm_clock_max_mhz'), columns):
        result[key] = max(values) if values else None
    result['available'] = bool(result['samples'])
    return result


def aggregate_round(rows, round_id):
    drafted = sum(row['drafted'] for row in rows)
    accepted = sum(row['accepted'] for row in rows)
    return dict(
        round=round_id, samples=len(rows),
        prefill_tps=sum(row['timings']['prompt_n'] for row in rows)
                    / (sum(row['timings']['prompt_ms'] for row in rows) / 1000),
        decode_tps=sum(row['timings']['predicted_n'] - 1 for row in rows)
                   / (sum(row['timings']['predicted_ms'] for row in rows) / 1000),
        ttft_ms=statistics.mean(row['ttft_ms'] for row in rows),
        wall_seconds=statistics.mean(row['wall_seconds'] for row in rows),
        drafted=drafted, accepted=accepted,
        acceptance_rate=accepted / drafted if drafted else None,
        no_draft_requests=sum(row['drafted'] == 0 for row in rows),
        matching_text_requests=sum(row['matches_reference_text'] for row in rows),
    )


def build_summary(folder):
    folder = Path(folder).resolve()
    require(folder.is_dir(), f'No results directory: {folder}')
    errors = sorted(path.name for path in folder.glob('*.error.json'))
    require(not errors, f'Failed runs in result directory: {errors}')
    expected = {f'{method}-r{round_id}.samples.jsonl'
                for method in METHODS for round_id in ROUNDS}
    actual = {path.name for path in folder.glob('*.samples.jsonl')}
    require(actual == expected,
            f'Incomplete/unexpected run matrix: missing={sorted(expected-actual)}, '
            f'extra={sorted(actual-expected)}')

    result = dict(
        schema_version=1, folder=str(folder),
        evaluation='Spec-Bench subset / llama.cpp / fixed 512 input and 256 output',
        rounds_per_method=5, unique_prompts=SAMPLE_COUNT, formal_requests=0,
        methods={}, telemetry={}, raw_sha256={},
        metric_definitions={
            'prefill_tps': 'sum(prompt_n) / sum(prompt_ms / 1000), within each round/scope',
            'decode_tps': 'sum(predicted_n - 1) / sum(predicted_ms / 1000), within each round/scope',
            'ttft_ms': 'mean client time to first nonempty text SSE, excludes model loading',
            'wall_seconds': 'mean client completion request wall time',
            'acceptance_rate': 'sum(draft_n_accepted) / sum(draft_n); null if no draft tokens',
            'paired_ratios': 'method metric / reference metric within the same round and scope',
            'matching_text': 'exact concatenated SSE content equality against same-round reference',
            'stdev': 'sample standard deviation of five independent round aggregates (ddof=1)',
        },
    )
    protocol_path = folder / 'protocol.json'
    result['protocol'] = read_json(protocol_path) if protocol_path.exists() else None
    for path in sorted(folder.iterdir()):
        if path.is_file():
            result['raw_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    canonical = None
    reference_texts = {}
    for method in METHODS:
        all_samples = []
        per_scope = {scope: [] for scope in SCOPES}
        for round_id in ROUNDS:
            label = f'{method}-r{round_id}'
            done = read_json(folder / f'{label}.done.json')
            require(done.get('samples') == SAMPLE_COUNT, f'{label}: done count != 12')
            prompts = read_json(folder / f'{label}.prompts.json')
            current = {key: prompts[key] for key in ('tokens', 'offsets')}
            require(len(current['tokens']) == SAMPLE_COUNT
                    and len(current['offsets']) == SAMPLE_COUNT, f'{label}: prompt count != 12')
            require(all(isinstance(tokens, list) and len(tokens) == 512
                        and all(isinstance(token, int) and not isinstance(token, bool)
                                and token >= 0 for token in tokens)
                        for tokens in current['tokens']), f'{label}: prompt must contain 512 token IDs')
            category_counts = Counter(item['category'] for item in current['offsets'])
            require(category_counts == Counter(dict(summarization=6, rag=6)),
                    f'{label}: expected six questions per category, got {category_counts}')
            identities = [(item['category'], item['question_id']) for item in current['offsets']]
            require(len(set(identities)) == SAMPLE_COUNT, f'{label}: duplicate question identity')
            if canonical is None:
                canonical = current
                result['inputs'] = prompts
            require(current == canonical, f'{label}: input tokens or metadata differ across runs')
            rows = [json.loads(line) for line in (folder / f'{label}.samples.jsonl')
                    .read_text(encoding='utf-8').splitlines() if line.strip()]
            require(len(rows) == SAMPLE_COUNT, f'{label}: actual sample count != 12')
            indexes = [row['prompt'] for row in rows]
            require(all(isinstance(index, int) and not isinstance(index, bool) for index in indexes)
                    and sorted(indexes) == list(range(SAMPLE_COUNT)), f'{label}: invalid prompt indexes')
            samples = []
            for row in sorted(rows, key=lambda value: value['prompt']):
                index = row['prompt']
                name = f'{label}/prompt{index}'
                timings, final = row['timings'], row['final']
                require(timings.get('prompt_n') == 512 and timings.get('predicted_n') == 256
                        and final.get('tokens_predicted') == 256 and timings.get('cache_n') == 0,
                        f'{name}: expected uncached 512/256 token accounting')
                for key in ('prompt_ms', 'predicted_ms'):
                    positive(timings.get(key), f'{name}/{key}')
                ttft = positive(row.get('ttft_seconds'), f'{name}/ttft_seconds')
                wall = positive(row.get('wall_seconds'), f'{name}/wall_seconds')
                require(ttft <= wall, f'{name}: TTFT exceeds completed request wall time')
                drafted = count(timings.get('draft_n', 0), f'{name}/draft_n')
                accepted = count(timings.get('draft_n_accepted', 0), f'{name}/draft_n_accepted')
                require(accepted <= drafted, f'{name}: accepted tokens exceed drafted tokens')
                require(method != 'reference' or drafted == 0, f'{name}: reference used drafting')
                output = returned_text(row, name)
                require(output != '', f'{name}: no returned text')
                if method == 'reference':
                    reference_texts[round_id, index] = output
                reference = reference_texts[round_id, index]
                samples.append(dict(
                    round=round_id, prompt=index, **canonical['offsets'][index],
                    timings=timings, ttft_ms=ttft * 1000, wall_seconds=wall,
                    drafted=drafted, accepted=accepted,
                    acceptance_rate=accepted / drafted if drafted else None,
                    output_text=output, output_sha256=text_hash(output),
                    reference_output_sha256=text_hash(reference),
                    matches_reference_text=output == reference,
                    first_differing_character=first_difference(output, reference),
                ))
            all_samples.extend(samples)
            for scope in SCOPES:
                selected = samples if scope == 'all' else [s for s in samples if s['category'] == scope]
                per_scope[scope].append(aggregate_round(selected, round_id))
            result['telemetry'][label] = telemetry(folder / f'{label}.gpu.csv')
        scopes = {}
        for scope, rounds in per_scope.items():
            drafted = sum(r['drafted'] for r in rounds)
            accepted = sum(r['accepted'] for r in rounds)
            scopes[scope] = dict(
                rounds=rounds, stats={key: stats([r[key] for r in rounds]) for key in METRICS},
                totals=dict(requests=sum(r['samples'] for r in rounds), drafted=drafted, accepted=accepted,
                            acceptance_rate=accepted / drafted if drafted else None,
                            no_draft_requests=sum(r['no_draft_requests'] for r in rounds),
                            matching_text_requests=sum(r['matching_text_requests'] for r in rounds)),
            )
        result['methods'][method] = dict(scopes=scopes, samples=all_samples)
        result['formal_requests'] += len(all_samples)
    for method, data in result['methods'].items():
        for scope, values in data['scopes'].items():
            references = result['methods']['reference']['scopes'][scope]['rounds']
            values['paired_ratios'] = {
                key: stats([row[key] / baseline[key] for row, baseline in zip(values['rounds'], references)])
                for key in METRICS
            }
    return result


def markdown_report(result, output_path, report_path):
    def link(path):
        return Path(os.path.relpath(path, report_path.parent)).as_posix()

    names = dict(reference='Q4 GPU 无投机', ngram='Q4 GPU + ngram-simple',
                 draft3='Q4 GPU + TinyLlama draft3')
    scopes = dict(all='两类合计', summarization='摘要', rag='RAG')
    lines = ['# Spec-Bench 子集：本机固定长度投机解码测试', '',
             '已完成三种方法各五轮：每轮相同的 12 个问题（摘要 6 个、RAG 6 个），共 180 条正式请求。'
             '每次单并发，输入 512、输出 256 token；预热不计入结果。五轮是重复测量，不是 60 个独立问题。', '',
             '数据来自 [Spec-Bench](https://github.com/hemingkx/Spec-Bench)。'
             '这是该数据子集在本项目合并模型与 llama.cpp 上的长度受控测试，不是官方完整 Spec-Bench 成绩。'
             '模型仍是原有 Llama 2 7B + 中文 LoRA 合并后的 Q4_K_M；Q4 GPU 无投机是本次投机对照，'
             '项目原始精度基线仍为 Merge BF16。', '',
             '## 性能结果', '',
             '以下为五轮均值 ± 样本标准差；每轮 prefill/decode 用该组总 token 数除以总耗时，'
             '不平均每题 token/s。摘要与 RAG 分别列出，合计对两类采用相同题数。', '',
             '| 任务 | 方法 | Prefill token/s ↑ | Decode token/s ↑ | TTFT ms ↓ | 请求耗时 s ↓ |',
             '|---|---|---:|---:|---:|---:|']
    for scope in SCOPES:
        for method in METHODS:
            values = result['methods'][method]['scopes'][scope]['stats']
            cells = [f"{values[key]['mean']:.2f} ± {values[key]['stdev']:.2f}" for key in METRICS]
            lines.append(f'| {scopes[scope]} | {names[method]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 同轮配对变化', '',
              '同一轮、同一任务下方法指标除以无投机对照，再对五个比值求均值。'
              '吞吐变化为正表示更快；TTFT/请求耗时变化为正表示更慢。', '',
              '| 任务 | 方法 | Prefill 变化 | Decode 变化 | TTFT 变化 | 请求耗时变化 |',
              '|---|---|---:|---:|---:|---:|']
    for scope in SCOPES:
        for method in METHODS[1:]:
            ratios = result['methods'][method]['scopes'][scope]['paired_ratios']
            cells = [f"{(ratios[key]['mean'] - 1) * 100:+.1f}%" for key in METRICS]
            lines.append(f'| {scopes[scope]} | {names[method]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 草稿接受与返回文本', '',
              '接受率 = 五轮接受草稿 token 总数 / 提出草稿 token 总数，不是回答正确率，'
              '也不是官方 Spec-Bench 的平均每次验证推进长度。零草稿请求保留在性能统计中；无草稿时接受率记为不适用。', '',
              '| 任务 | 方法 | 接受/提出 token | 接受率 | 无草稿请求 | 返回文本与同轮对照完全相同 |',
              '|---|---|---:|---:|---:|---:|']
    for scope in SCOPES:
        for method in METHODS:
            total = result['methods'][method]['scopes'][scope]['totals']
            rate = '不适用' if total['acceptance_rate'] is None else f"{100 * total['acceptance_rate']:.2f}%"
            lines.append(f"| {scopes[scope]} | {names[method]} | {total['accepted']}/{total['drafted']} | "
                         f"{rate} | {total['no_draft_requests']}/{total['requests']} | "
                         f"{total['matching_text_requests']}/{total['requests']} |")
    lines += ['', '比较对象为所有 SSE content 拼接后的原始返回文本，不裁剪空白或 EOS，不根据部分流式 token ID '
              '宣称逐 token 一致。逐请求文本、SHA256、是否不同及首个不同字符位置保存在 JSON 摘要中。', '',
              '## 合计逐轮与范围', '',
              '| 方法 | 轮次 | Prefill token/s | Decode token/s | TTFT ms | 请求耗时 s |',
              '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for row in result['methods'][method]['scopes']['all']['rounds']:
            lines.append(f"| {names[method]} | {row['round']} | "
                         + ' | '.join(f'{row[key]:.3f}' for key in METRICS) + ' |')
    lines += ['', '| 方法 | Prefill 范围 | Decode 范围 | TTFT ms 范围 | 请求耗时 s 范围 |',
              '|---|---:|---:|---:|---:|']
    for method in METHODS:
        values = result['methods'][method]['scopes']['all']['stats']
        lines.append(f'| {names[method]} | ' + ' | '.join(
            f"{values[key]['min']:.3f}–{values[key]['max']:.3f}" for key in METRICS) + ' |')
    lines += ['', '## 输入与运行证据', '', '| 任务 | question_id | 最终输入 token |', '|---|---|---:|']
    for item in result['inputs']['offsets']:
        lines.append(f"| {scopes[item['category']]} | {item['question_id']} | 512 |")
    lines += ['', f"[结构化摘要及所有逐轮/逐题数据]({link(output_path)})，"
              f"[首轮输入 token 与来源元数据]({link(Path(result['folder']) / 'reference-r1.prompts.json')})。"
              '摘要记录全部原始文件 SHA256；运行目录内保留命令、协议、SSE、完成状态和 GPU 采样。', '',
              '| 方法 | GPU 显存采样最大值 MiB | 温度采样最大值 °C |', '|---|---:|---:|']
    for method in METHODS:
        telemetry_rows = [result['telemetry'][f'{method}-r{r}'] for r in ROUNDS]
        cells = []
        for key in ('whole_gpu_memory_max_mib', 'temperature_max_c'):
            values = [row[key] for row in telemetry_rows if row[key] is not None]
            cells.append(f'{max(values):.1f}' if values else '未获得')
        lines.append(f'| {names[method]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 口径与限制', '',
              '- 摘要与 RAG 是偏输入重叠的长上下文任务，未覆盖官方其余四类，也不能代表中文问答能力。'
              '样本按运行前确定的规则选取；512 长度预算下仅裁原有上下文，不用重复问题或无关文字填充。'
              '裁上下文会改变原任务并可能丢失证据，不能把本次输出当官方质量分数。',
              '- 请求关闭提示缓存并忽略 EOS，固定产生 256 token；重复、复制或本可提前结束的回答也保留。'
              '结果描述此负载下的性能，尚未完成正式质量/PPL 验收，接受率或文本相同不等于质量通过。',
              '- Decode 计时按 llama.cpp 的口径使用每请求 255 个计时 token；TTFT 是客户端收到首个非空文本 '
              'SSE 的时间，含请求和传输开销，不含模型加载。官方 Spec-Bench 完整调用 wall-time 吞吐与该 decode 指标不同。',
              '- GPU 是整卡定时采样，包含桌面等其他占用，不是进程精确峰值；温度、频率、功耗与热漂移会影响结果。'
              '五轮统计只描述本次测量波动，不能消除小样本与运行环境限制。', '',
              '固定数据版本和取样、命令见 [实验协议](specbench-subset.md)。'
              '官方协议参考：[评测循环](https://github.com/hemingkx/Spec-Bench/blob/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/evaluation/eval.py)、'
              '[速度统计](https://github.com/hemingkx/Spec-Bench/blob/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/evaluation/speed.py)。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    output, report = args.output.resolve(), args.report.resolve()
    require(output != report, 'JSON and Markdown outputs must differ')
    require(not output.is_relative_to(folder) and not report.is_relative_to(folder),
            'Generated summaries must stay outside the raw result directory')
    result = build_summary(folder)
    markdown = markdown_report(result, output, report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report.write_text(markdown, encoding='utf-8')
    print(f"Validated {result['formal_requests']} requests, 12 unique prompts, 5 rounds per method")
    for method in METHODS:
        values = result['methods'][method]['scopes']['all']['stats']
        print(method, {key: round(values[key]['mean'], 3) for key in METRICS})


if __name__ == '__main__':
    main()

"""Select long-context Spec-Bench inputs before observing model outputs."""
import hashlib
import json

REVISION = 'fd2c1cd7d2201ef71db4c5f4e455008f017967bf'
SOURCE = f'https://raw.githubusercontent.com/hemingkx/Spec-Bench/{REVISION}/data/spec_bench/question.jsonl'


def build_inputs(raw, tokenize, detokenize, per_task=6):
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    selected = {}
    audit = []
    suffix = tokenize('\n\n### Response:\n')
    for category in ['summarization', 'rag']:
        eligible = []
        for row in rows:
            if row['category'] != category:
                continue
            assert len(row['turns']) == 1
            original = row['turns'][0]
            if category == 'summarization':
                if not original.startswith('Summarize: '):
                    raise ValueError('Unexpected summarization structure')
                instruction, context = 'Summarize:', original[len('Summarize: '):]
            else:
                context, instruction = original.rsplit('\n', 1)
                if not instruction.strip():
                    raise ValueError('Missing RAG question')
            prefix_text = '### Instruction:\n' + instruction + '\n\n### Input:\n'
            prefix = tokenize(prefix_text, True)
            context_tokens = tokenize(context)
            budget = 512 - len(prefix) - len(suffix)
            valid = budget > 0 and len(context_tokens) >= budget
            rank = hashlib.sha256(f'specbench-v1:42:{category}:{row["question_id"]}'.encode()).hexdigest()
            metadata = dict(category=category, question_id=row['question_id'],
                original_context_tokens=len(context_tokens), context_tokens=budget,
                rank_sha256=rank, source_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest())
            audit.append(dict(**metadata, eligible=valid))
            if valid:
                tokens = prefix + context_tokens[:budget] + suffix
                assert len(tokens) == 512
                eligible.append(dict(metadata=metadata, tokens=tokens,
                    instruction=instruction, context=detokenize(context_tokens[:budget]),
                    readable_prompt=detokenize(tokens)))
        eligible.sort(key=lambda r:r['metadata']['rank_sha256'])
        if len(eligible) < per_task + 1:
            raise ValueError(f'Not enough eligible {category} inputs: {len(eligible)}')
        selected[category] = eligible[:per_task+1]
    # Interleave tasks; hold out another real document for excluded warmup.
    samples = [selected[category][i] for i in range(per_task) for category in ['summarization','rag']]
    warmup = selected['summarization'][per_task]
    return dict(source=SOURCE, revision=REVISION, corpus_sha256=hashlib.sha256(raw).hexdigest(),
        selection='SHA256(specbench-v1:42:category:question_id), smallest eligible per task; interleaved',
        tokens=[r['tokens'] for r in samples], offsets=[r['metadata'] for r in samples],
        readable_inputs=samples, eligibility=audit,
        warmup_offset=warmup['metadata'], warmup_tokens=warmup['tokens'])

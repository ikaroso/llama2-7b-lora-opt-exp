"""Protect paired comparisons against incomplete runs and changed workloads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_specbench import METHODS, ROUNDS, aggregate_round, build_summary, markdown_report


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class ReportSpecBenchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.prompts = dict(tokens=[[1] + [100 + i] * 511 for i in range(12)],
                            offsets=[dict(category='summarization' if i < 6 else 'rag',
                                          question_id=1000 + i) for i in range(12)])
        for method in METHODS:
            for round_id in ROUNDS:
                label = f'{method}-r{round_id}'
                write(self.folder / f'{label}.done.json', dict(samples=12))
                write(self.folder / f'{label}.prompts.json', self.prompts)
                rows = []
                for i in range(12):
                    timings = dict(prompt_n=512, predicted_n=256, cache_n=0,
                                   prompt_ms=100 * (i + 1), predicted_ms=2550 * (i + 1))
                    if method != 'reference' and i != 0:
                        timings.update(draft_n=100, draft_n_accepted=50)
                    final = dict(stop=True, content='', tokens_predicted=256, timings=timings)
                    content = f'answer {i}'
                    if method == 'ngram' and i == 2:
                        content += ' different'
                    # Deliberately partial streaming token IDs; these must not be
                    # mistaken for the complete 256 generated-token sequence.
                    rows.append(dict(prompt=i, repeat=0, timings=timings, final=final,
                                     ttft_seconds=(i + 1) / 10, wall_seconds=10 * (i + 1),
                                     events=[dict(data=dict(content=content, tokens=[42])),
                                             dict(data=final)]))
                self.write_rows(label, rows)

    def write_rows(self, label, rows):
        (self.folder / f'{label}.samples.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')

    def read_rows(self, label):
        return [json.loads(line) for line in (self.folder / f'{label}.samples.jsonl')
                .read_text(encoding='utf-8').splitlines()]

    def test_aggregate_uses_total_time_not_average_tps(self):
        rows = [dict(timings=dict(prompt_n=512, prompt_ms=100, predicted_n=256, predicted_ms=1000),
                     drafted=100, accepted=20, ttft_ms=120, wall_seconds=1.2,
                     matches_reference_text=True),
                dict(timings=dict(prompt_n=512, prompt_ms=300, predicted_n=256, predicted_ms=3000),
                     drafted=900, accepted=720, ttft_ms=320, wall_seconds=3.3,
                     matches_reference_text=False)]
        result = aggregate_round(rows, 1)
        self.assertEqual(result['prefill_tps'], 2560)
        self.assertEqual(result['decode_tps'], 127.5)
        self.assertEqual(result['acceptance_rate'], .74)
        self.assertEqual(result['ttft_ms'], 220)

    def test_complete_matrix_text_comparison_and_category_stats(self):
        result = build_summary(self.folder)
        self.assertEqual(result['formal_requests'], 180)
        ngram = result['methods']['ngram']
        self.assertEqual(ngram['scopes']['all']['totals']['matching_text_requests'], 55)
        self.assertEqual(ngram['scopes']['all']['totals']['no_draft_requests'], 5)
        self.assertEqual(ngram['scopes']['all']['totals']['acceptance_rate'], .5)
        self.assertEqual(ngram['scopes']['rag']['totals']['requests'], 30)
        self.assertEqual(ngram['scopes']['all']['paired_ratios']['decode_tps']['values'], [1] * 5)
        self.assertEqual(ngram['samples'][2]['first_differing_character'], len('answer 2'))
        self.assertIsNone(result['methods']['reference']['scopes']['all']['totals']['acceptance_rate'])
        self.assertEqual(len(result['raw_sha256']), 45)
        report = markdown_report(result, self.folder.parent / 'summary.json', self.folder.parent / 'report.md')
        self.assertIn('180 条正式请求', report)
        self.assertIn('不是官方完整 Spec-Bench 成绩', report)

    def test_missing_round_fails(self):
        (self.folder / 'draft3-r5.samples.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'run matrix'):
            build_summary(self.folder)

    def test_error_or_incomplete_done_fails(self):
        write(self.folder / 'ngram-r2.done.json', dict(samples=11))
        with self.assertRaisesRegex(ValueError, 'done count'):
            build_summary(self.folder)
        write(self.folder / 'ngram-r2.done.json', dict(samples=12))
        write(self.folder / 'reference-r1.error.json', dict(error='interrupted'))
        with self.assertRaisesRegex(ValueError, 'Failed runs'):
            build_summary(self.folder)

    def test_cache_or_token_accounting_fails(self):
        for key, value in [('cache_n', 1), ('prompt_n', 511), ('predicted_n', 255)]:
            with self.subTest(key=key):
                rows = self.read_rows('reference-r1')
                original = copy.deepcopy(rows)
                rows[0]['timings'][key] = value
                self.write_rows('reference-r1', rows)
                with self.assertRaisesRegex(ValueError, 'uncached 512/256'):
                    build_summary(self.folder)
                self.write_rows('reference-r1', original)

    def test_changed_prompt_or_metadata_fails(self):
        for key in ('tokens', 'offsets'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.prompts)
                if key == 'tokens':
                    changed['tokens'][0][-1] += 1
                else:
                    changed['offsets'][0]['question_id'] += 100
                write(self.folder / 'draft3-r5.prompts.json', changed)
                with self.assertRaisesRegex(ValueError, 'differ across runs'):
                    build_summary(self.folder)
                write(self.folder / 'draft3-r5.prompts.json', self.prompts)

    def test_duplicate_request_index_fails(self):
        rows = self.read_rows('ngram-r1')
        rows[-1]['prompt'] = 0
        self.write_rows('ngram-r1', rows)
        with self.assertRaisesRegex(ValueError, 'prompt indexes'):
            build_summary(self.folder)

    def test_impossible_acceptance_fails(self):
        rows = self.read_rows('draft3-r1')
        rows[1]['timings']['draft_n_accepted'] = 101
        self.write_rows('draft3-r1', rows)
        with self.assertRaisesRegex(ValueError, 'exceed drafted'):
            build_summary(self.folder)


if __name__ == '__main__':
    unittest.main()

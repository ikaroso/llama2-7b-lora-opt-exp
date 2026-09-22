import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from specbench_inputs import build_inputs


class InputsTests(unittest.TestCase):
    def test_keeps_question_selects_without_reference_and_no_padding(self):
        rows=[]
        for category in ['summarization','rag']:
            for i in range(4):
                text=('Summarize: '+('a'*900)) if category=='summarization' else ('b'*900+'\nWho is named?')
                rows.append(dict(question_id=len(rows),category=category,turns=[text],reference=['SECRET']))
        rows.append(dict(question_id=99,category='rag',turns=['short\nShort question?']))
        def tokenize(text,special=False): return ([1] if special else [])+[ord(c)+2 for c in text]
        def detokenize(tokens): return ''.join(chr(t-2) if t!=1 else '<s>' for t in tokens)
        raw='\n'.join(json.dumps(r) for r in rows).encode()
        result=build_inputs(raw,tokenize,detokenize,per_task=2)
        self.assertEqual([r['category'] for r in result['offsets']],['summarization','rag']*2)
        self.assertTrue(all(len(t)==512 for t in result['tokens']))
        self.assertNotIn(result['warmup_offset']['question_id'],[r['question_id'] for r in result['offsets']])
        self.assertFalse(next(r for r in result['eligibility'] if r['question_id']==99)['eligible'])
        for row in result['readable_inputs']:
            self.assertNotIn('SECRET',row['readable_prompt'])
            self.assertIn(row['instruction'],row['readable_prompt'])
            self.assertTrue(row['readable_prompt'].endswith('### Response:\n'))
        self.assertEqual(result,build_inputs(raw,tokenize,detokenize,per_task=2))


if __name__=='__main__': unittest.main()

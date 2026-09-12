"""Check fixed token accounting and SSE completion detection without running a model."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import optimize


def response(prompt_n=512, predicted=256, complete=True):
    rows=[dict(content='', tokens=[1], stop=False), dict(content='text', tokens=[2], stop=False)]
    if complete:
        rows.append(dict(stop=True,timings=dict(prompt_n=prompt_n),tokens_predicted=predicted))
    return io.BytesIO(b''.join(b'data: '+json.dumps(r).encode()+b'\n\n' for r in rows))


class StreamTests(unittest.TestCase):
    def test_sse_and_budget(self):
        with patch.object(optimize,'post',return_value=response()) as post:
            result=optimize.stream(0,[1]*512)
            self.assertIsNotNone(result['ttft_seconds'])
            self.assertEqual(len(result['events']),3)
            self.assertFalse(post.call_args.args[2]['cache_prompt'])
            self.assertEqual(post.call_args.args[2]['n_predict'],256)

    def test_reject_wrong_budget(self):
        for pn,gn in [(511,256),(512,255)]:
            with patch.object(optimize,'post',return_value=response(pn,gn)):
                with self.assertRaises(RuntimeError): optimize.stream(0,[1]*512)

    def test_reject_incomplete_stream(self):
        with patch.object(optimize,'post',return_value=response(complete=False)):
            with self.assertRaises(RuntimeError): optimize.stream(0,[1]*512)


if __name__=='__main__': unittest.main()

"""Offline tests for the user's one-request constraint; no network calls."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import requests
from industrial_swap_v2.core.qwen_client import generate, payload


class FakeContext:
    def __init__(self, root):
        self.root = Path(root)
        self.records = self.root / 'records'
        self.records.mkdir()
        self.cfg = {'m2': {'model':'qwen-image-2.1-pro','candidates':1,'max_api_requests':1,
                          'output_size':'1024*1024','seed':42,'prompt_extend':False,'timeout_seconds':600}}
    def stage(self, _):
        path = self.root / 'm2'
        path.mkdir(exist_ok=True)
        return path
    def receipt(self, *_args, **_kwargs):
        pass


class FakeResponse:
    status_code = 200
    def json(self):
        return {'output':{'choices':[{'finish_reason':'stop','message':{'content':[{'image':'https://example.invalid/result.png'}]}}]},'request_id':'offline-test'}
    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, timeout=False):
        self.posts = 0
        self.gets = 0
        self.timeout = timeout
    def post(self, *_args, **kwargs):
        self.posts += 1
        assert kwargs['allow_redirects'] is False
        assert kwargs['json']['parameters']['n'] == 1
        assert 'negative_prompt' not in kwargs['json']['parameters']
        if self.timeout:
            raise requests.Timeout()
        return FakeResponse()
    def get(self, *_args, **_kwargs):
        self.gets += 1
        response = FakeResponse()
        image = io.BytesIO()
        Image.new('RGB',(16,16)).save(image,format='PNG')
        response.content = image.getvalue()
        return response


class OnceTests(unittest.TestCase):
    @patch('industrial_swap_v2.core.qwen_client.api_key', return_value='offline-placeholder')
    def test_cached_response_and_download_resume_do_not_repost(self, _):
        with tempfile.TemporaryDirectory() as directory:
            ctx = FakeContext(directory)
            path = Path(directory)/'input.png'
            Image.new('RGB',(16,16)).save(path)
            session = FakeSession()
            output = generate(ctx,[path],'replace device',session)
            generate(ctx,[path],'replace device',session)
            output.unlink()
            generate(ctx,[path],'replace device',session)
            self.assertEqual((session.posts,session.gets),(1,2))
            with self.assertRaises(RuntimeError):
                generate(ctx,[path],'a changed prompt',session)
            self.assertEqual(session.posts,1)
    @patch('industrial_swap_v2.core.qwen_client.api_key', return_value='offline-placeholder')
    def test_timeout_consumes_claim_without_resubmission(self, _):
        with tempfile.TemporaryDirectory() as directory:
            ctx = FakeContext(directory)
            path = Path(directory)/'input.png'
            Image.new('RGB',(16,16)).save(path)
            session = FakeSession(timeout=True)
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    generate(ctx,[path],'replace device',session)
            self.assertEqual(session.posts,1)


if __name__ == '__main__':
    unittest.main()

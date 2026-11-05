import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from industrial_swap_v2.core.credentials import api_key

class CredentialTests(unittest.TestCase):
    @patch.dict(os.environ, {'DASHSCOPE_API_KEY':''})
    def test_complete_dotted_key_is_never_truncated(self):
        key = 'sk-ws-v.example.region.abcdefghijklmnopqrstuvwxyz_0123456789-ABCDEFG'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'credential.txt'
            for text in (key, key+'\n', '\ufeff'+key, 'DASHSCOPE_API_KEY='+key,
                         'export DASHSCOPE_API_KEY="'+key+'"'):
                path.write_text(text,encoding='utf-8')
                self.assertEqual(api_key(path),key)
            path.write_text('label: '+key,encoding='utf-8')
            with self.assertRaises(ValueError):
                api_key(path)

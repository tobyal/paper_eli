"""Credential-shape diagnostics only; makes no API call and prints no key."""
import os
from pathlib import Path
import itertools
from industrial_swap_v2.core.context import Context, arguments
from industrial_swap_v2.core.credentials import api_key, inspect
ctx = Context(arguments().config)
path = ctx.cfg['m2']['api_key_file']
key = api_key(path)
print({'selected_source': 'environment' if os.environ.get('DASHSCOPE_API_KEY','').strip() else 'configured_file',
       'key_length':len(key), 'key_has_sk_prefix':key.startswith('sk-'),
       'file_metadata':inspect(path),
       'character_class_runs': [[kind,len(list(items))] for kind,items in itertools.groupby(
           ''.join('A' if c.isascii() and c.isalnum() else 'W' if c.isspace() else 'U' if not c.isascii() else c
                   for c in Path(path).read_text(encoding='utf-8-sig')))]},flush=True)

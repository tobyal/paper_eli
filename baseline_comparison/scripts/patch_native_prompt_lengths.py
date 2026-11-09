"""Respect the frozen CLIP encoder's 77-token capacity; retain full input text."""
from pathlib import Path
import difflib
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';patch=[]
for repo in [ROOT/'repos/DGE',BASE/'vendor/GaussianEditor']:
 p=repo/'threestudio/models/prompt_processors/stable_diffusion_prompt_processor.py';t=p.read_text();u=t
 for tokenizer in ['self.tokenizer','tokenizer']:
  u=u.replace(f'max_length={tokenizer}.model_max_length,\n',f'max_length={tokenizer}.model_max_length,\n                truncation=True,\n')
 needle='        with torch.no_grad():\n            tokens = tokenizer('
 u=u.replace(needle,'''        lengths = [len(tokenizer(text, truncation=False).input_ids) for text in prompts]
        for text, length in zip(prompts, lengths):
            if length > tokenizer.model_max_length:
                print(f"NATIVE_CLIP_TOKEN_LIMIT input={length} used={tokenizer.model_max_length}: {text}", flush=True)
        with torch.no_grad():
            tokens = tokenizer(''',1)
 assert u!=t
 p.write_text(u);patch.extend(difflib.unified_diff(t.splitlines(True),u.splitlines(True),fromfile=str(p)+' before',tofile=str(p)+' after'))
(BASE/'records/native_clip_token_capacity.patch').write_text(''.join(patch));print('Native CLIP token-length compatibility recorded')

"""Explicit human authorization only; never called by the pipeline or retry logic."""
import argparse
import shutil
from pathlib import Path
from industrial_swap_v2.core.context import Context, read_json, write_json

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--authorization', required=True)
a = p.parse_args()
ctx = Context(a.config)
if a.authorization not in ('已更新 Key，允许再试一次', '允许再请求一次'):
    raise ValueError('An explicit renewed user authorization is required')
status = read_json(ctx.records/'qwen_request_status.json')
if status['status'] != 'http_error_no_retry':
    raise ValueError('Cannot release a submitted, ambiguous or successful request')
lock = ctx.records/'qwen_generation_once.lock'
number = len(list(ctx.records.glob('qwen_attempt_*.lock'))) + 1
archived = ctx.records/f'qwen_attempt_{number:03d}.lock'
if archived.exists():
    raise FileExistsError(archived)
shutil.copy2(ctx.stage('m2')/'api_request.json', ctx.records/f'qwen_attempt_{number:03d}_request.json')
write_json(ctx.records/f'qwen_attempt_{number:03d}_status.json', status)
write_json(ctx.records/f'qwen_authorization_{number+1:03d}.json', {
    'user_authorization': a.authorization, 'additional_submissions': 1, 'images': 1,
    'previous_attempt_status': status, 'automatic_retry': False})
lock.rename(archived)
print('Archived unsuccessful attempt; one explicitly authorized submission is available')

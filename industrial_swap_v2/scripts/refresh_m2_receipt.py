"""Migrate completed M2 accounting to distinguish failed attempts from images."""
from industrial_swap_v2.core.context import Context, arguments, read_json, write_json
ctx=Context(arguments().config)
receipt=read_json(ctx.records/'m2_status.json')
assert receipt['status']=='complete'
record=read_json(ctx.stage('m2')/'generation.json')
attempts=1+len(list(ctx.records.glob('qwen_attempt_*.lock')))
record.pop('generation_requests',None)
record.update(submission_attempts=attempts,successful_generation_requests=1)
write_json(ctx.stage('m2')/'generation.json',record)
ctx.complete('m2',[item['path'] for item in receipt['artifacts']],submission_attempts=attempts,successful_generation_requests=1)
print('Updated accounting only; no API call, no image regeneration')

"""Add Chinese descriptions of visually inspected results; no automatic quality claims."""
import json,os
from pathlib import Path
root=Path(os.environ['ROOT'])
descriptions={
 'gaussianeditor':'完整运行2500步。五个检查视角中的目标底座仍主要为白色，未明显达到改红目标；可见轻微颜色与细节变化。',
 'dge':'完整运行1000步。五个检查视角中的目标仍主要为白色，部分区域出现紫蓝色偏色和边缘伪影，未明显达到改红目标。',
 'dreamcatalyst':'完整运行1500步。五个检查视角中的目标仍主要为白色，目标附近出现较明显的绿色/紫色边缘及表面伪影，未达到改红目标。',
}
for method,description in descriptions.items():
 path=root/f'records/{method}.quality_review.json'
 review=json.loads(path.read_text())
 assert review['quality_label']=='execution_success_edit_target_not_achieved'
 review['visual_result_zh']=description
 path.write_text(json.dumps(review,ensure_ascii=False,indent=2))
print('Added Chinese descriptions for the three inspected completed methods')

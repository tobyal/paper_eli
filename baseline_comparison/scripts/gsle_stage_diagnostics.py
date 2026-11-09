"""Expose native GSLE reference/coarse/refinement artifacts without rerendering."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageOps,ImageFont
BASE=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison');records={}
font=ImageFont.truetype('DejaVuSans.ttf',16)
for case in ['case00','case02','case04','case07','case10']:
 folder=BASE/'gsle'/case
 paths=[folder/'foreground/reference_white.png',folder/'coarse/previews/reference_006500.png',folder/'refine/previews/reference_002000.png']
 if not all(p.exists() for p in paths):continue
 board=Image.new('RGB',(1536,610),'white');draw=ImageDraw.Draw(board)
 draw.text((8,6),case+' | GSLE native anchor diagnostics (not the common comparison cameras)',fill='black',font=font)
 labels=['Generated foreground reference','Coarse 6500 steps','DI-SDS 2000 steps'];metrics={}
 for i,(path,label) in enumerate(zip(paths,labels)):
  draw.text((512*i+8,34),label,fill='black',font=font)
  with Image.open(path) as source:im=ImageOps.contain(source.convert('RGB'),(512,512))
  board.paste(im,(512*i+(512-im.width)//2,64+(512-im.height)//2))
  if i:
   stage=['','coarse','refine'][i];progress=json.loads((folder/'records'/(stage+'_progress.json')).read_text())
   metrics[stage]={k:progress[k] for k in ['step','rgb_mse','mask_mse','sds_loss','points']}
   draw.text((512*i+8,584),f'Own-reference RGB MSE {progress["rgb_mse"]:.5f} | Mask MSE {progress["mask_mse"]:.5f}',fill='black',font=font)
 board.save(BASE/'comparison'/case/'gsle_stages.png',dpi=(300,300));board.save(BASE/'comparison'/case/'gsle_stages.jpg',quality=93)
 records[case]=metrics
(BASE/'comparison/gsle_stage_metrics.json').write_text(json.dumps({'metric_scope':'Native losses against each case\'s own generated foreground. Not a cross-method/cross-case quality score; SDS values from different diffusion teachers/stages are not directly comparable.','cases':records},indent=2))
print('Saved native GSLE stage diagnostics for',len(records),'cases')

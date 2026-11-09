"""Publication-sized primary-camera montage, using existing identical target crops."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli/baseline_comparison/comparison')
cases=[('case00','Yellow six-axis robot arm'),('case02','Roller conveyor'),('case04','Process chiller'),('case07','Parts-storage cabinet'),('case10','Tool cabinet')]
out=Image.new('RGB',(3000,5*680),'white');draw=ImageDraw.Draw(out);font=ImageFont.truetype('DejaVuSans.ttf',24)
for i,(case,label) in enumerate(cases):
 assert (ROOT/case/'gsle/artifact.json').exists(), 'GSLE final comparison is not ready'
 with Image.open(ROOT/case/'comparison_crops.png') as im:row=im.crop((0,0,3000,640))
 draw.text((10,i*680+8),case+' | '+label,fill='black',font=font);out.paste(row,(0,i*680+40))
out.save(ROOT/'overview_primary.png',dpi=(300,300))
ImageOps.contain(out,(2500,3000)).save(ROOT/'overview_primary.jpg',quality=95)
print('Saved five-case primary-camera overview')

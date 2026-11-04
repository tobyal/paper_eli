"""Create a labeled comparison from actual saved renders, without synthesis."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

root=Path(os.environ['ROOT'])
dest=root/'outputs/comparison'
dest.mkdir(parents=True,exist_ok=True)
methods=['reconstruction','gaussianeditor','dge','dreamcatalyst','editsplat']
labels=['Unedited 3DGS','GaussianEditor','DGE','DreamCatalyst','EditSplat']
frames=[61,91,151,161,281]
heldout=set(json.loads((root/'records/camera_split.json').read_text())['test'])
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
def read(method,name):
    return Image.open(root/'outputs'/method/'common_render/wide'/name).convert('RGB')
def target_box(name):
    mask=np.array(Image.open(root/'outputs/evaluation_masks/wide'/name))>127
    y,x=np.where(mask)
    assert len(x)>100, str(name)
    cx,cy=(x.min()+x.max())/2,(y.min()+y.max())/2
    side=max(x.max()-x.min()+1,y.max()-y.min()+1)*1.45
    side=max(side,90)
    return (max(0,round(cx-side/2)),max(0,round(cy-side/2)),min(mask.shape[1],round(cx+side/2)),min(mask.shape[0],round(cy+side/2)))
sheet=Image.new('RGB',(320*5,350*len(frames)+45),'white')
draw=ImageDraw.Draw(sheet)
for j,label in enumerate(labels): draw.text((j*320+10,12),label,font=font,fill='black')
for i,frame in enumerate(frames):
    name=f'frame_{frame:06d}.png'
    box=target_box(name)
    for j,method in enumerate(methods):
        im=ImageOps.contain(read(method,name).crop(box),(320,320),Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(320,320),(235,235,235))
        tile.paste(im,((320-im.width)//2,(320-im.height)//2))
        sheet.paste(tile,(j*320,i*350+45))
        split_label='test' if Path(name).with_suffix('.jpg').name in heldout else 'train'
        draw.text((j*320+8,i*350+369),f'Camera {frame:03d} / {(frame-1)/2:.1f}s / {split_label}',font=small,fill='black')
sheet.save(dest/'target_comparison.png')

frames_dir=dest/'video_frames'
frames_dir.mkdir(exist_ok=True)
source=list(sorted((root/'outputs/reconstruction/common_render/wide').glob('*.png')))
for i,path in enumerate(source):
    grid=Image.new('RGB',(1920,1132),'black')
    draw=ImageDraw.Draw(grid)
    for j,method in enumerate(methods[1:]):
        x,y=(j%2)*960,(j//2)*566
        grid.paste(read(method,path.name),(x,y+26))
        draw.text((x+10,y+3),labels[j+1],font=font,fill='white')
    grid.save(frames_dir/f'{i:06d}.jpg',quality=95)
subprocess.run(['ffmpeg','-hide_banner','-loglevel','warning','-y','-framerate','12','-i',str(frames_dir/'%06d.jpg'),'-c:v','libx264','-threads','8','-crf','18','-pix_fmt','yuv420p',str(dest/'four_methods_multiview.mp4')],check=True)
(dest/'comparison_manifest.json').write_text(json.dumps({'methods':methods,'target_crop_frames':frames,'source_frames':len(source),'video_fps':12,'capture_fps':2,'video_speedup':6,'images':'Actual saved renders; same camera and same crop for each method; no pixel recoloring or generated image substitution'},indent=2))
print(dest,flush=True)

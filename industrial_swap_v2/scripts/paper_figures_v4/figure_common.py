"""Data provenance and publication plots; distinct from GSLE's common module."""
import csv,hashlib,json,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
BASE=ROOT/'industrial_swap_v2'
OUT=BASE/'outputs/paper_figures_v4'
OUT.mkdir(parents=True,exist_ok=True)
CASES=json.loads((ROOT/'baseline_comparison/prompts.json').read_text())['cases']
COLORS=['#225E9D','#D57A27','#399276','#9A5599','#747B83']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,s):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def source(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def csvout(p,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def folder(name):
    p=OUT/name;p.mkdir(parents=True,exist_ok=True);(p/'panels').mkdir(exist_ok=True);return p
def export(fig,d,key,caption,sources,rows,axes=None):
    # Fixed layout, then crop each panel from the same vector figure coordinates.
    fig.canvas.draw();fig.savefig(d/f'figure_{key}.pdf',facecolor='white')
    fig.savefig(d/f'figure_{key}.png',dpi=300,facecolor='white')
    if axes is not None:
        renderer=fig.canvas.get_renderer()
        for i,ax in enumerate(axes):
            box=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025,1.04)
            fig.savefig(d/'panels'/f'{chr(97+i)}.png',bbox_inches=box,dpi=300,facecolor='white')
    plt.close(fig)
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(d/f'figure_{key}.pdf'),str(d/f'pdf_preview_{key}')],check=True)
    Image.open(d/f'pdf_preview_{key}.png').convert('RGB').save(d/f'preview_{key}.jpg',quality=94)
    (d/'caption.txt').write_text(caption+'\n',encoding='utf-8');save(d/'sources.json',sources);csvout(d/'data.csv',rows)
    print(key,'COMPLETE',d,flush=True)
def m4folder(row):
    if row['case']=='case07':return Path(row['ours_source'])/'m4'
    return Path(row['ours_scene']).parent

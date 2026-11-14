"""F08: actual saved localization + deduplicated preliminary instance audit."""
import csv, hashlib, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT=Path(os.environ.get('ROOT',str(Path(__file__).resolve().parents[3])))
BASE=ROOT/'industrial_swap_v2/outputs/paper_figures_v4'
SRC=BASE/'F08_localization_revised'
OUT=BASE/'F08_localization_final'

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True);(OUT/'panels').mkdir(exist_ok=True);(OUT/'code').mkdir(exist_ok=True)
    audit=read(Path(__file__).with_name('f08_instance_audit.json'))
    index=read(SRC/'case_index.json')['cases'];byid={x['m1_case_id']:x for x in index}
    groups={}
    for x in audit['cases']:groups.setdefault(x['physical_target_group'],[]).append(x)
    assert len(groups)==9
    records=[]
    for group,rows in groups.items():
        for method in ['text_only','photo_assisted']:
            assert len({r[method+'_correct'] for r in rows})==1
        r=rows[0];rec=byid[r['m1_case_id']]
        for row in rows:
            source=SRC/'cases'/row['m1_case_id']
            assert (source/'photo_assisted_indices.npy').exists() and (source/'text_only_indices.npy').exists()
        records.append(dict(physical_target_id=group,m1_labels=';'.join(f"M1-{x['number']:02d}" for x in rows),
            m1_case_ids=';'.join(x['m1_case_id'] for x in rows),description=rec['description'],
            text_only_correct=int(r['text_only_correct']),photo_guided_correct=int(r['photo_assisted_correct']),
            audit_status='Preliminary AI Visual Audit; independent human adjudication pending',
            evidence=';'.join(x['notes'] for x in rows)))
    # Corroborate the duplicate grouping directly from immutable memberships.
    ids04=np.load(SRC/'cases'/index[3]['m1_case_id']/'photo_assisted_indices.npy')
    ids09=np.load(SRC/'cases'/index[8]['m1_case_id']/'photo_assisted_indices.npy')
    duplicate_coverage=len(np.intersect1d(ids04,ids09))/min(len(ids04),len(ids09))
    assert duplicate_coverage>.98
    with (OUT/'instance_selection.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    counts=[sum(r[k] for r in records) for k in ['text_only_correct','photo_guided_correct']]
    rec=next(x for x in index if x['catalog_number']==10);source=SRC/'cases'/rec['m1_case_id']
    mask=np.asarray(Image.open(source/'reference_mask.png'))>127
    yy,xx=np.where(mask);crop=[max(0,int(xx.min())-28),max(0,int(yy.min())-28),min(mask.shape[1],int(xx.max())+29),min(mask.shape[0],int(yy.max())+29)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':8,
        'axes.labelsize':7,'xtick.labelsize':6.6,'ytick.labelsize':6.6,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(180/25.4,63/25.4),facecolor='white');axes=[]
    for i,(method,label) in enumerate([('text_only','Text-only Localization'),('photo_assisted','Photo-guided Localization')]):
        ax=fig.add_axes([.009+i*.373,.17,.36,.73]);axes.append(ax)
        im=Image.open(source/'view_0'/(method+'_highlight.png'))
        ax.imshow(im);ax.contour(mask,levels=[.5],colors=['#00cfe7'],linewidths=.6);ax.axis('off')
        ax.set_title(f'({chr(97+i)}) {label}',loc='left',pad=5)
        zoom=ax.inset_axes([.64,.58,.35,.39]);zoom.imshow(im);zoom.contour(mask,levels=[.5],colors=['#00cfe7'],linewidths=.7)
        zoom.set_xlim(crop[0],crop[2]);zoom.set_ylim(crop[3],crop[1]);zoom.set_xticks([]);zoom.set_yticks([])
        zoom.set_title('Intended target',fontsize=5.3,pad=1.5)
        if i==0:
            ins=ax.inset_axes([.01,.01,.40,.43]);ins.imshow(Image.open(source/'native_text/text_only_highlight.png'))
            ins.set_xticks([]);ins.set_yticks([]);ins.set_title('Selected elsewhere',fontsize=5.3,pad=1.5)
    ax=fig.add_axes([.817,.24,.171,.57]);axes.append(ax)
    ax.bar([0,1],np.array(counts)/len(records),width=.55,color=['#c58439','#226b99'],zorder=3)
    for i,n in enumerate(counts):ax.text(i,n/9+.045,f'{n}/9',ha='center',fontsize=8,fontweight='bold')
    ax.set_ylim(0,1.21);ax.set_yticks([0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([0,1],['Text-only','Photo-guided']);ax.tick_params(axis='x',labelsize=6.1)
    ax.grid(axis='y',color='#eeeeee',zorder=0);ax.set_title('(c) Correct Instance\nSelection',loc='left',pad=5)
    fig.legend(handles=[Line2D([0],[0],color='#e34c46',lw=4,label='Selected Gaussians'),
        Line2D([0],[0],color='#00cfe7',lw=1.4,label='Intended target')],loc='center',bbox_to_anchor=(.37,.10),ncol=2,frameon=False,fontsize=6.6)
    fig.savefig(OUT/'figure_F08.png',dpi=300);fig.savefig(OUT/'figure_F08.pdf')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for i,a in enumerate(axes):
        box=a.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.07)
        fig.savefig(OUT/'panels'/f'{chr(97+i)}.png',bbox_inches=box,dpi=300)
    plt.close(fig)
    caption=('Fig. 8. Reference-image-guided instance localization in industrial scenes. '
        '(a-b) The accepted M1-10 comparison uses the same registered main camera: red denotes actual selected Gaussians and cyan the user-designated target. '
        'Insets show the intended target and the text-only selection at its actual location. '
        f'(c) Correct-instance counts are {counts[0]}/9 for text-only and {counts[1]}/9 for photo-guided LangSplat after grouping M1-04 and M1-09 as the same physical cooler. '
        'Preliminary AI Visual Audit: these judgments combine saved multi-view evidence and existing user-confirmed target identities; independent human adjudication remains pending. '
        'Both settings use the same per-task text and frozen LangSplat fields. Some saved descriptions, including M1-10, imperfectly describe the selected equipment, so description mismatch is a confound in addition to instance ambiguity. '
        'All nine targets belong to one industrial scene. No supplied simulated IoU table, new generation, geometry change or camera adjustment is used.')
    (OUT/'caption.txt').write_text(caption+'\n')
    evidence=[Path(__file__).with_name('f08_instance_audit.json'),SRC/'case_index.json',source/'reference_mask.png',
        source/'view_0/text_only_highlight.png',source/'view_0/photo_assisted_highlight.png',source/'native_text/text_only_highlight.png']
    write(OUT/'sources.json',dict(source_directory=str(SRC),audit_type=audit['reviewer_type'],human_adjudicated=False,
        physical_targets=9,counts=counts,duplicate_smaller_set_coverage=duplicate_coverage,
        main_camera=rec['reference_camera'],inset_camera=rec['text_native_camera'],main_crop=None,target_detail_crop=crop,
        real_gaussian_projections=True,simulated_table_used=False,figure_mm=[180,63],dpi=300,
        inputs=[dict(path=str(p),sha256=sha(p)) for p in evidence]))
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('f08_instance_audit.json'),OUT/'code/f08_instance_audit.json')
    subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(OUT/'figure_F08.pdf'),str(OUT/'pdf_proof')],check=True)
    print('F08 completed',counts,'/ 9',flush=True)

if __name__=='__main__':main()

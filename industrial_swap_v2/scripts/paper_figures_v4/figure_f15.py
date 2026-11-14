import sqlite3,re
from datetime import datetime
import numpy as np
from figure_common import *

def main():
    d=folder('F15_efficiency');rows=[];details=[];provenance=[]
    db=sqlite3.connect('file:'+str(BASE/'webapp/data/database/app.db')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    batch_path=BASE/'outputs/qwen_reselected_04_07_10_m4_2000_v1/batch.json';batch={x['key']:x for x in read(batch_path)['items']}
    for row in CASES:
        case=row['case'];base=Path(row['ours_source']);m3=base/'m3';m4=m4folder(row);m1=Path(row['target_record']).parents[2]
        m1_seconds=None;job_record=None
        if (m1/'case_source.json').exists():
            case_dir=Path(read(m1/'case_source.json')['folder']);jid=read(case_dir/'confirmed_case.json')['job_id']
            rec=db.execute('SELECT id,status,started_at,finished_at FROM m1_jobs WHERE id=?',(jid,)).fetchone()
            if rec and rec['started_at'] and rec['finished_at']:
                job_record=dict(rec);m1_seconds=(datetime.fromisoformat(rec['finished_at'])-datetime.fromisoformat(rec['started_at'])).total_seconds()
        trellis_path=m3/'trellis_generation.json'
        if not trellis_path.exists():trellis_path=Path(read(m3/'asset_provenance.json')['source_glb']).parent/'trellis_generation.json'
        trellis=read(trellis_path)['inference_seconds'];train=read(m3/'pbr_training_summary.json')['seconds']
        alignment=read(m4/'alignment/report.json')['seconds'];optimization=read(m4/'background_optimization/summary.json')['seconds']
        extraction=batch.get(case,{}).get('stages',{}).get('m2',{}).get('seconds')
        m4_summary=read(m4/'summary.json');guard=bool(m4_summary['candidate_guard_passed'])
        record=dict(case=case,M1_Time_s=m1_seconds if m1_seconds is not None else 'NA',M2_Time_s='NA',M3_Time_s='NA',M4_Time_s='NA',Total_Time_s='NA',
            Guard_Status='passed' if guard else 'candidate_failed_guard',Guarded_Selected_Step=m4_summary['selected_step'],
            M2_import_segmentation_s=extraction if extraction is not None else 'NA',TRELLIS_inference_s=trellis,PBR_training_s=train,
            Registration_s=alignment,Background_optimization_s=optimization,
            Measured_subtotal_s=sum(x for x in [m1_seconds,extraction,trellis,train,alignment,optimization] if x is not None),
            Missing_components='Qwen/manual generation; PBR view rendering/export; M4 supervision preparation/evaluation overhead'+('; initial localization' if case=='case00' else '')+('; final manual yaw adjustment' if case=='case07' else ''))
        rows.append(record)
        for component,value in [('M1 localization job',m1_seconds),('M2 import + SAM',extraction),('M3 TRELLIS inference',trellis),('M3 Gaussian training',train),('M4 registration',alignment),('M4 background optimization',optimization)]:
            if value is not None:details.append(dict(case=case,component=component,seconds=value))
        provenance.append(dict(case=case,m1_database_record=job_record,trellis=source(trellis_path),pbr=source(m3/'pbr_training_summary.json'),
            alignment=source(m4/'alignment/report.json'),background=source(m4/'background_optimization/summary.json'),guard=source(m4/'summary.json'),
            m2_import_timing_source=str(batch_path) if extraction is not None else None))
    reconstruction_log=ROOT/'logs/05_reconstruction.time.txt';text=reconstruction_log.read_text();duration=re.search(r'Elapsed.*?: (\d+):(\d+\.\d+)',text)
    reconstruction_seconds=int(duration.group(1))*60+float(duration.group(2)) if duration else None
    fig=plt.figure(figsize=(9,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],left=.19,right=.98,bottom=.14,top=.88,wspace=.5,hspace=.55)
    fig.suptitle('Computational Efficiency: Recorded Runtime Components',y=.97,fontsize=13,fontweight='bold')
    ax=fig.add_subplot(gs[0,:]);x=np.arange(5);left=np.zeros(5);groups=[('M1 localization job',[r['M1_Time_s'] for r in rows]),('M2 import + SAM',[r['M2_import_segmentation_s'] for r in rows]),('M3 inference + training',[r['TRELLIS_inference_s']+r['PBR_training_s'] for r in rows]),('M4 registration + optimization',[r['Registration_s']+r['Background_optimization_s'] for r in rows])]
    for i,(label,values) in enumerate(groups):
        values=np.array([float(v)/60 if v!='NA' else 0 for v in values]);ax.barh(x,values,left=left,color=COLORS[i],label=label);left+=values
    ax.set_yticks(x,[r['case'] for r in rows]);ax.invert_yaxis();ax.set(xlabel='Recorded component time (minutes)',title='(a) Measured runtime breakdown - incomplete end-to-end coverage')
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.21),fontsize=7);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    b=fig.add_subplot(gs[1,0]);names=['M1 localization job','M2 import + SAM','M3 TRELLIS inference','M3 Gaussian training','M4 registration','M4 background optimization'];short=['M1 query/job','M2 import/SAM','TRELLIS','Gaussian train','Registration','Background opt.']
    for i,name in enumerate(names):
        vals=np.array([x['seconds']/60 for x in details if x['component']==name]);b.errorbar(vals.mean(),i,xerr=[[vals.mean()-vals.min()],[vals.max()-vals.mean()]],fmt='o',color=COLORS[0],capsize=3)
        b.text(vals.max()+.15,i,f'n={len(vals)}',va='center',fontsize=7)
    b.set_yticks(range(len(names)),short);b.invert_yaxis();b.set(xlabel='Minutes: mean and observed range',title='(b) Recorded computational costs');b.grid(axis='x',alpha=.18)
    cc=fig.add_subplot(gs[1,1]);cc.bar(range(5),[r['Measured_subtotal_s']/60 for r in rows],color='white',edgecolor=COLORS[0],hatch='///')
    cc.set_xticks(range(5),['00','02','04','07','10']);cc.set(xlabel='Case',ylabel='Recorded subtotal (minutes)',title='(c) Partial runtime totals');cc.grid(axis='y',alpha=.18);cc.set_axisbelow(True)
    fig.text(.08,.025,f'Unknown times are missing, never zero. Subtotals are lower bounds, not complete pipeline runtimes.\nOne-time 30k-iteration source 3DGS training: {reconstruction_seconds/60:.2f} min (COLMAP/LangSplat excluded). Shared GPUs.',fontsize=8)
    caption='Figure 15. Computational efficiency extracted from original run records, without rerunning the pipeline. (a) Available measured components grouped by stage. (b) Means and observed ranges, with the number of recorded cases. (c) Recorded-time subtotals, explicitly lower bounds. End-to-end M2 generation times are unavailable for manually supplied images; PBR view-rendering/export and M4 supervision overhead are not consistently instrumented. Therefore complete stage/total times are marked NA in T03 rather than inferred from file modification times or treated as zero. TRELLIS inference and Gaussian training times are counted separately, including cached asset provenance. Case00 has no original localization-job duration; its later cache-import time is excluded. Case07 timings precede the accepted manual yaw correction. The one-time source-scene 3DGS training duration is reported separately; CPU RSS is not reported as GPU memory.'
    export(fig,d,'F15',caption,dict(cases=provenance,reconstruction=source(reconstruction_log),reconstruction_seconds=reconstruction_seconds,missing_is_not_zero=True),rows,[ax,b,cc])
    csvout(d/'stage_components.csv',details);tables=OUT/'tables';tables.mkdir(exist_ok=True);csvout(tables/'T03_efficiency.csv',rows)

if __name__=='__main__':main()

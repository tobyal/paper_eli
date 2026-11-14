"""Publish only measured values; explicit NA for independent success/attribute labels."""
import numpy as np,csv
from figure_common import *

def tex_escape(x):
    return str(x).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')

def table(caption,columns,rows,label):
    body=['\\begin{table*}[t]','\\centering','\\small',r'\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+'l'*len(columns)+'}',r'\toprule',
        ' & '.join(tex_escape(x) for x in columns)+r' \\',r'\midrule']
    body+=[' & '.join(tex_escape(x) for x in row)+r' \\' for row in rows]
    return '\n'.join(body+[r'\bottomrule',r'\end{tabular}',r'\end{table*}'])

def main():
    d=OUT/'tables';d.mkdir(exist_ok=True);rows=[]
    def add(module,variant,n,metric,value,notes='',case='aggregate'):
        rows.append(dict(module=module,variant=variant,case=case,n=n,metric=metric,value=float(value),notes=notes))
    loc=read(OUT/'F08_localization/localization_metrics.json')
    for method,label in [('gsle','Text-only'),('ours','Photo-assisted')]:
        a=[r for r in loc if r['method']==method]
        for key in ['reference_sam_iou','reference_sam_recall','reference_sam_precision']:add('M1',label,len(a),key,np.mean([r[key] for r in a]),'Input-reference agreement, not held-out accuracy')
    for r in read(OUT/'F11_m2_ablation/metrics.json'):
        for key in ['object_inside_edit_fraction','background_rgb_mae','background_psnr','clip_text_relevance']:add('M2',r['condition'],1,key,r[key],'One paired-seed sample; ROI-SAM location estimate; CLIP is not accuracy',case='case00')
    for stage in ['baseline','best']:
        a=[read(Path(r['ours_source'])/'m3/pbr_training_summary.json')[stage]['validation'] for r in CASES]
        for key in ['foreground_psnr','depth_mae_relative','silhouette_iou']:add('M3','Initial' if stage=='baseline' else 'RGB+Depth',5,key,np.mean([x[key] for x in a]),'Held-out PBR views; no RGB-only run')
    alignment=read(OUT/'F13_spatial_alignment/ablation_results.json')
    for method in ['A_bbox','B_silhouette','C_full']:
        a=[r for r in alignment if r['method']==method]
        for key in ['mask_iou','center_offset_fraction','support_outside_fraction']:add('M4 alignment',method,len(a),key,np.mean([r[key] for r in a]),'Fixed Case00/02 assets; center offset is a prior deviation, not pose error')
    base=Path(CASES[0]['ours_source'])/'m4';baseline=read(base/'background_optimization/baseline.json')['summary']['validation']
    full=read(base/'summary.json');rgb=read(OUT/'F14_background_refinement/rgb_only_run/m4/background_optimization/optimization_log.json')['evaluations'][-1]
    for name,data in [('Step 0',baseline),('RGB only candidate',rgb['summary']['validation']),('Full M4 candidate',full['candidate']['validation']),('Guarded final',full['guarded_final']['validation'])]:
        for key in ['environment_l1','trusted_visibility']:add('M4 background',name,1,key,data[key],'2000-step candidates fail guard; guarded export selects Step 0',case='case00')
    csvout(d/'T02_ablation.csv',rows)
    overall=read(d/'overall_metrics.json');aggregate=[]
    for method in ['gaussianeditor','dge','GSLE (coarse)','ours']:
        a=[r for r in overall if r['method']==method];out=dict(method=method,n=len(a),Localization_IoU='NA',Replacement_Success_Rate='NA',Text_Attribute_Accuracy='NA')
        for key in ['Reference_SAM_IoU','Workspace_Overflow_OpacityWeightedCenters','Background_LPIPS','Background_RGB_MAE']:
            v=[r[key] for r in a if r[key]!='NA'];out[key]=float(np.mean(v)) if v else 'NA'
        aggregate.append(out)
    csvout(d/'T01_aggregate.csv',aggregate)
    fmt=lambda x:f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    t1=table('Overall comparison over five cases. Ref-SAM is input-reference agreement, not independent localization ground truth. Center-out is opacity-weighted Gaussian centers outside the intended shared workstation, not volumetric overflow. LPIPS uses three shared cameras and the same excluded region per case. Low background change alone does not indicate successful replacement. Success/attribute accuracy and independently annotated IoU are not evaluated. Ours uses the existing guard-rejected 2000-step candidates (Case07 retains its accepted yaw adjustment); GSLE uses coarse, matching F07.',
        ['Method','Ref-SAM IoU','Center-out','BG LPIPS','GT IoU','Success','Attributes'],
        [[{'gaussianeditor':'GaussianEditor','dge':'DGE','ours':'Ours'}.get(r['method'],r['method']),fmt(r['Reference_SAM_IoU']),fmt(r['Workspace_Overflow_OpacityWeightedCenters']),fmt(r['Background_LPIPS']),'NA','NA','NA'] for r in aggregate],'tab:overall')
    selected=[]
    for r in rows:
        if r['metric'] in ['reference_sam_iou','object_inside_edit_fraction','background_rgb_mae','foreground_psnr','depth_mae_relative','mask_iou','environment_l1','trusted_visibility']:
            display={'reference_sam_iou':'Input-SAM IoU','object_inside_edit_fraction':'Inside-edit fraction','background_rgb_mae':'BG RGB MAE','foreground_psnr':'FG PSNR (dB)','depth_mae_relative':'Depth MAE / diagonal','mask_iou':'Silhouette IoU','environment_l1':'Environment L1','trusted_visibility':'Trusted visibility'}
            selected.append([r['module'],r['variant'],r['n'],display[r['metric']],fmt(r['value'])])
    t2=table('Actual module ablations. Metrics have different units and evaluation domains, defined in the corresponding figure captions. M1 is an in-sample SAM diagnostic; M2 has one sample per condition; M3 uses held-out rendered views. Unrun RGB-only M3 and independent success evaluations are omitted. M4 candidates are separated from the guard-selected export.',
        ['Module','Variant','n','Metric','Value'],selected,'tab:ablation')
    runtime=list(csv.DictReader((d/'T03_efficiency.csv').open()))
    t3=table('Recorded engineering performance (seconds). Missing complete-stage times are NA, not zero. Recorded subtotal is a lower bound comprising available components, not end-to-end latency. Qwen/manual input generation, PBR view export and some supervision overhead were not consistently timed. Exact components and source logs are in the CSV. All five 2000-step candidates failed at least one preservation guard; guarded exports selected step 0.',
        ['Case','M1','M2','M3','M4','Total','Recorded subtotal','Guard'],
        [[r['case'],fmt(float(r['M1_Time_s'])) if r['M1_Time_s']!='NA' else 'NA','NA','NA','NA','NA',f"{float(r['Measured_subtotal_s']):.1f}",'Failed; step 0'] for r in runtime],'tab:runtime')
    (d/'tables.tex').write_text('% Requires \\usepackage{booktabs}. NA = not evaluated or not recorded.\n\n'+'\n\n'.join([t1,t2,t3])+'\n',encoding='utf-8')
    save(d/'table_sources.json',dict(F08=str(OUT/'F08_localization/sources.json'),F11=str(OUT/'F11_m2_ablation/sources.json'),F12=str(OUT/'F12_m3_training/sources.json'),F13=str(OUT/'F13_spatial_alignment/sources.json'),F14=str(OUT/'F14_background_refinement/sources.json'),F15=str(OUT/'F15_efficiency/sources.json'),T01=str(d/'overall_sources.json'),nonfinite_values=False))
    print('TABLES COMPLETE',flush=True)

if __name__=='__main__':main()

"""Publication layout of the accepted v8 figures; no camera/model changes."""
import shutil
from PIL import Image
from layout import OUT,ROOT,read,save,source,grid,sha


def main():
    dest=OUT/'F07_comparison';assert not dest.exists();dest.mkdir()
    source_root=ROOT/'baseline_comparison/comparison_updated_v8'
    cases=['case00','case02','case04','case07','case10']
    methods=['gaussianeditor','dge','gsle','ours'];labels=['GaussianEditor','DGE','GSLE Coarse','Ours']
    rows=[];row_labels=[];records={}
    for n,case in enumerate(cases):
        folder=dest/'panels'/case;folder.mkdir(parents=True)
        record=read(source_root/case/'report.json');cams=read(source_root/case/'camera_params.json')
        files={}
        for stage in ['localization','final']:
            paths=[]
            for m in methods:
                p=source_root/case/f'{m}_{stage}.png';out=folder/p.name;shutil.copy2(p,out)
                assert sha(out)==sha(p);paths.append(out);files[p.name]=source(p)
                v=cams['methods'][m].get('views',cams['methods'][m])['main']
                assert v['localization_camera']==v['final_camera']
            rows.append(paths);row_labels.append(f'({chr(97+n)}) Case{case[4:]}\n'+('Localization' if stage=='localization' else 'Replacement'))
        # Retain the accepted detailed crops, auxiliary views and per-case layout.
        shutil.copy2(source_root/case/'comparison_paired.png',dest/f'{case}.png')
        for name in ['comparison_crops.png','comparison_aux1.png','comparison_aux2.png']:
            shutil.copy2(source_root/case/name,folder/name)
        records[case]=dict(input_files=files,camera_params=cams,display_stage='GSLE Coarse 6500, before DI-SDS',
            prior_report=str(source_root/case/'report.json'),source_gaussians=record['methods']['gsle']['final_scene'],
            all_method_source_records=record['methods'])
    layout=grid(dest/'figure_F07','Qualitative Comparison with Existing 3D Gaussian Editing Methods',
        rows,column_titles=labels,row_titles=row_labels,left=67,fontsize=7.5,title_fontsize=9.,
        footer='Within each method: identical localization/result camera. GSLE shows Coarse 6500, before DI-SDS.')
    caption='Qualitative comparison of GaussianEditor, DGE, GSLE Coarse, and our method on five industrial equipment replacement cases. Each case contains localization (top) and replacement (bottom) rows. The translucent red annotation represents the original Gaussians actually selected by each method. Each method uses identical cameras for localization and replacement; the accepted cross-method framing is preserved, and different methods may use different world-space cameras. GSLE is explicitly shown at the saved Coarse 6500-step stage, before DI-SDS refinement, rather than its full refined output. GaussianEditor, DGE, and Ours use their previously selected completed scene outputs. Localization inputs differ: the baselines use their native localization, while Ours additionally uses an original reference photograph to specify the target instance. The figure is a viewpoint-matched qualitative comparison, not a pixel-aligned evaluation under identical localization inputs. All panels are rendered experimental outputs; failures and real occlusion are retained.'
    (dest/'caption.txt').write_text(caption,encoding='utf-8')
    save(dest/'comparison_sources.json',dict(accepted_version=str(source_root),cases=records,layout=layout,
        camera_changes=0,training_steps=0,new_baseline_runs=0,gsle_stage='coarse',original_panels_byte_identical=True))
    print('F07 COMPLETE',dest,flush=True)


if __name__=='__main__':main()

"""SCI layout of existing, read-only Gaussian renders; run with system Python."""
import argparse
from layout import OUT, grid, read, save


def f05():
    d=OUT/'F05_case00';p=d/'panels'
    rows=[[p/'a_registered_reference.png',p/'b_target_localization.png',p/'c_qwen_reference.png'],
          [p/'d_original_3dgs.png',p/'e_reconfigured_scene.png',p/'f_novel_view.png']]
    labels=[['(a) Registered Reference\nImage','(b) Target Localization','(c) Text-Guided\n2D Reference'],
            ['(d) Original 3DGS Scene','(e) Reconfigured 3D Scene','(f) Novel-View Result']]
    meta=grid(d/'figure_F05','Industrial Equipment Replacement in a Real Factory Scene',rows,
        cell_labels=labels,fontsize=7.5,title_fontsize=9.,
        footer='Case00 | (d) and (e): identical camera. All 3D views render the saved Gaussian scenes.')
    (d/'caption.txt').write_text('Figure 5. Industrial equipment replacement in a real factory scene. '
        '(a) The registered factory photograph used for reference-assisted localization, with the selected SAM region highlighted. '
        '(b) The selected original-scene Gaussians, shown with a translucent red annotation. '
        '(c) The existing Qwen-generated scene reference for a yellow six-axis industrial robot. '
        '(d) The original 3DGS scene and (e) the integrated replacement scene, rendered with exactly the same camera. '
        '(f) A novel view of the same final scene. The final result uses the PBR-trained object and the saved '
        '2,000-step local-background optimization result; no scene parameters were changed for this figure. '
        'The registered photograph in (a) is not a Gaussian render.\n',encoding='utf-8')
    s=read(d/'sources.json');s['layout']=meta;save(d/'sources.json',s)
    print('F05 COMPLETE',flush=True)


def f06():
    d=OUT/'F06_five_cases';cases=['case00','case02','case04','case07','case10']
    rows=[[d/'panels'/x/(k+'.png') for k in ['original','m2_reference','m3_gaussian_asset','m4_integrated_scene']] for x in cases]
    meta=grid(d/'figure_F06','Stage-wise Results of Text-Guided Industrial Equipment Replacement',rows,
        column_titles=['(a) Original\nScene','(b) M2: Text-Guided\nReference','(c) M3: Gaussian\nAsset','(d) M4: Integrated\nScene'],
        row_titles=['Case00\nSix-axis robot','Case02\nRoller\nconveyor','Case04\nChiller','Case07\nStorage\ncabinet','Case10\nTool cabinet'],
        left=65,fontsize=7.1,title_fontsize=8.8,
        footer='Original and integrated scenes use the same camera within each row. Assets: white-background Gaussian renders.')
    (d/'caption.txt').write_text('Figure 6. Stage-wise results for five industrial equipment replacement cases. '
        'Columns show (a) the original industrial 3DGS scene, (b) the existing Qwen scene reference, '
        '(c) the independent Gaussian asset trained using PBR multi-view supervision, and '
        '(d) the final scene after spatial registration and local-background optimization. '
        'Rows correspond to a six-axis robot (Case00), roller conveyor (Case02), chiller (Case04), '
        'automated storage cabinet (Case07), and tool cabinet (Case10). '
        'Within each row, the original and integrated scenes use identical M2 Anchor camera parameters. '
        'Independent Gaussian assets are rendered on white backgrounds with comparable framing; '
        'only their presentation cameras differ. Case07 retains the accepted yaw adjustment, and each '
        'final PLY version is recorded in the accompanying manifest. No generation or optimization was performed for this figure.\n',encoding='utf-8')
    s=read(d/'case_manifest.json');s['layout']=meta;save(d/'case_manifest.json',s)
    print('F06 COMPLETE',flush=True)


def f10():
    d=OUT/'F10_multiview';records=read(d/'cameras.json')['views']
    rows=[[d/k/(x['view']+'.png') for x in records] for k in ['full_scene','equipment_detail']]
    meta=grid(d/'figure_F10','Multi-view Results of the Reconfigured Industrial Scene',rows,
        column_titles=['(a) Front\nView','(b) Left\nOblique View','(c) Right-side\nView','(d) Rear\nView','(e) Elevated\nView'],
        row_titles=['Full\nIndustrial\nScene','Equipment\nDetail'],left=52,fontsize=7.2,title_fontsize=9.,
        footer='Same saved Case00 scene as Figure 5. Bottom row: equal-size crops of the corresponding top-row renders.')
    (d/'caption.txt').write_text('Figure 10. Multi-view renders of the reconfigured industrial scene for Case00. '
        '(a) The front reference direction, (b) a left-oblique direction, (c) a right-side direction, '
        '(d) the rear direction, and (e) an elevated direction. The upper row shows the scene context; '
        'the lower row consists of equally sized pixel crops from the corresponding upper-row renders, '
        'so each column has exactly the same camera and occlusion relationships. '
        'All views use the identical final Gaussian scene used in Figure 5. '
        'Real foreground occlusion and incomplete reconstruction of unobserved factory surfaces are retained; '
        'no geometry, opacity, or colors were changed to improve presentation.\n',encoding='utf-8')
    s=read(d/'sources.json');s['layout']=meta;save(d/'sources.json',s)
    print('F10 COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--figure',choices=['F05','F06','F10'],required=True)
    a=p.parse_args();{'F05':f05,'F06':f06,'F10':f10}[a.figure]()

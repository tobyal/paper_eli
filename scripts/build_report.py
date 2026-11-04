"""Summarize measured artifacts and quality reviews, never infer semantic success."""
import json,os
from pathlib import Path
root=Path(os.environ['ROOT'])
methods=['gaussianeditor','dge','dreamcatalyst','editsplat']
labels={'gaussianeditor':'GaussianEditor','dge':'DGE','dreamcatalyst':'DreamCatalyst','editsplat':'EditSplat'}
metrics=json.loads((root/'records/common_metrics.json').read_text())
experiment=json.loads((root/'records/experiment.json').read_text())
rows=[]
for method in methods:
 artifact=json.loads((root/f'outputs/{method}/artifact_manifest.json').read_text())
 run=json.loads(Path(artifact['run_record']).read_text())
 quality=json.loads((root/f'records/{method}.quality_review.json').read_text())
 score=metrics['methods'][method]
 views=score['visible_target_view_aggregate'];pairs=score['pair_aggregate']
 def mean(d,k):return d.get(k,{}).get('mean')
 model_check=json.loads(Path(artifact['model_check']).read_text())
 rows.append({'method':labels[method],'run':run,'artifacts':artifact,'model':model_check,'quality':quality,
 'metrics':{'visible_target_test_views':views['target_red_fraction']['n'],'red_fraction':mean(views,'target_red_fraction'),
 'clip_target_similarity':mean(views,'clip_target_similarity'),
 'clip_directional_similarity':mean(views,'clip_directional_similarity'),
 'background_psnr_db':mean(views,'background_psnr_db'),
 'background_ssim':mean(views,'background_ssim'),'background_lpips':mean(views,'background_lpips'),
 'target_reprojection_mae':mean(pairs,'target_reprojection_mae'),
 'target_reprojection_excess_mae':mean(pairs,'target_reprojection_excess_mae'),
 'wall_minutes':run['wall_seconds']/60,'peak_process_gpu_gib':run['sampled_process_gpu_peak_mib']/1024}})
baseline=metrics['methods']['reconstruction']
baseline_red=baseline['visible_target_view_aggregate']['target_red_fraction']['mean']
baseline_reprojection=baseline['pair_aggregate']['target_reprojection_mae']['mean']
report={'instruction':experiment['instruction_zh'],'instruction_en':experiment['instruction_en'],'rows':rows,
 'unedited_reference':{'red_fraction':baseline_red,'target_reprojection_mae':baseline_reprojection},
 'limits':['仅使用一个工业视频、一个固定随机种子；没有编辑后的真实参考图像。',
 '共享GPU环境。耗时含模型加载、分割、编辑、拟合和方法自带测试；统一评价渲染的耗时单独记录。',
 '单次运行耗时不含共同预处理、基础3DGS训练、环境安装、模型下载及失败重试；相关日志全部另存。',
 '进程显存每5秒采样，报告的是采样峰值，可能漏掉短时峰值。',
 '统一使用源场景几何和人工检查过的近似目标mask。红色比例和CLIP是代理指标，不能单独证明编辑正确。',
 '较低的跨视角MAE也可能来自模糊或几乎没有编辑，必须结合实际RGB和编辑强度判断。',
 '背景LPIPS对空间特征图按mask汇总；其感受野可能受目标附近变化影响，应结合像素PSNR/SSIM和实际图像解读。',
 '共用306个训练视角、44个留出测试视角；各方法保留官方的视角采样策略。',
 '方法使用的视角数和优化次数不同，墙钟时间反映本配置的完整运行成本，不代表同等计算预算下的独立速度排名。',
 '目标中心相机实验仅作诊断，未用于主对比。']}
(root/'records/final_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
lines=['工业场景四方法复现实验',f'统一指令：{experiment["instruction_zh"]}',f'英文执行指令：{experiment["instruction_en"]}',
 f'服务器目录：{root}',
 '执行状态：四种官方方法均完成实际运行、输出检查和同相机渲染。',
 '编辑目标：本轮四种方法均未明显将指定外壳改红。此交付是实际运行的负结果基线，不能称为改色成功。' if all(r['quality']['quality_label']=='execution_success_edit_target_not_achieved' for r in rows) else '编辑目标达成情况以各方法的实际RGB检查记录为准。', '',
 '实际输出及指标（目标可见的24个测试视角；重投影为18组有效视角对）：',
 '方法 | 目标红色比例% | 背景PSNR dB | 背景SSIM | 背景LPIPS | 目标重投影MAE | 耗时分钟 | 采样显存峰值GiB']
for row in rows:
 s=row['metrics']
 lines.append(f'{row["method"]} | {100*s["red_fraction"]:.3f} | {s["background_psnr_db"]:.3f} | {s["background_ssim"]:.4f} | {s["background_lpips"]:.4f} | {s["target_reprojection_mae"]:.4f} | {s["wall_minutes"]:.2f} | {s["peak_process_gpu_gib"]:.2f}')
lines+=['','输出检查：']
for row in rows:
 lines += [row['method']+': '+row['quality'].get('visual_result_zh',row['quality']['visual_result']), 'PLY: '+row['artifacts']['ply'], 'Video: '+row['artifacts']['multiview_video'], 'Log: '+row['run']['log'],'']
best_psnr=max(rows,key=lambda r:r['metrics']['background_psnr_db'])
best_lpips=min(rows,key=lambda r:r['metrics']['background_lpips'])
fastest=min(rows,key=lambda r:r['metrics']['wall_minutes'])
slowest=max(rows,key=lambda r:r['metrics']['wall_minutes'])
lines+=['本配置下的比较：',
 f'未编辑模型的目标红色比例为{100*baseline_red:.3f}%，目标重投影MAE为{baseline_reprojection:.4f}，为编辑后数值提供参照。',
 f'背景像素保持指标中，{best_psnr["method"]}的PSNR最高；感知保持指标中，{best_lpips["method"]}的LPIPS最低。这些分数只衡量偏离原场景的程度，几乎不编辑也可能获得较好的保持分数。',
 f'{fastest["method"]}本次完整运行耗时最短（{fastest["metrics"]["wall_minutes"]:.2f}分钟），{slowest["method"]}最长（{slowest["metrics"]["wall_minutes"]:.2f}分钟）。各自使用官方采样/迭代设置，且共享服务器负载不恒定，不能解释为等预算速度排名。',
 '跨视角一致性使用源几何重投影，不是对优化后几何准确度的独立检验；请同时检查目标纹理、伪影和展示视频。','']
lines+=['诊断记录：',
 '官方IP2P在原始视图与目标中心视图上的诊断未稳定改红目标；部分结果误改前景工作台、地面标线或厂房背景。相关提示词、随机种子和真实二维图像均保存。',
 '这一视频包含多台相似设备，目标在不少编辑视角中不可见。二维实例/部件定位问题已在优化前出现，是解释本轮结果的重要证据。',
 'EditSplat参考视角151的MFG结果恢复了误改红的地面标线，但目标底座仍主要为白色。此处是中间二维诊断，不能替代最终3D指标。']
diagnostic=root/'records/editsplat_2d_diagnostics.json'
if diagnostic.exists():
 selected=json.loads(diagnostic.read_text())['stages']['initial_2d']['summary']['selected']
 lines.append(f'EditSplat初始筛选保留{selected["views"]}个视角，其中{selected["target_visible_views"]}个在共同目标mask中有至少100像素；这{selected["target_visible_views"]}个目标区域的平均红色比例为{100*selected["mean_target_red_fraction"]:.3f}%。全部{selected["views"]}个入选视角的目标外平均红色比例为{100*selected["mean_outside_target_red_fraction"]:.3f}%。')
lines+=['以上结果仅对应这一视频、指令和配置，不能推广为四篇论文在其他数据上的性能结论。','']
lines+=['解释与边界：']+report['limits']
lines+=['','复现：bash scripts/replay_baseline.sh repeat1',
 '环境安装、数据预处理和补丁应用顺序见 README_reproduction.txt。',
 '运行成功与改红成功分别记录。所有对比图像均为保存的真实RGB渲染；红色语义标注图不计作编辑结果。']
(root/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

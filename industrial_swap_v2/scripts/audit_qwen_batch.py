"""Audit existing batch artifacts and export a portable preview bundle; no GPU work."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import zipfile

from industrial_swap_v2.core.context import read_json, write_json
from industrial_swap_v2.webapp import store as s, service as v


def vertex_count(path):
    with Path(path).open('rb') as stream:
        for _ in range(1024):
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    raise ValueError('No vertex count: ' + str(path))


def probe(path):
    value = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,nb_read_frames:format=duration',
        '-of', 'json', str(path)], text=True))
    assert int(value['streams'][0]['nb_read_frames']) > 0
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    batch = args.batch.resolve()
    batch.relative_to(s.BASE / 'outputs')
    manifest = read_json(batch / 'final_manifest.json')
    if not args.allow_incomplete:
        assert len(manifest['items']) == manifest['completed_cases'] == 9
    report = {'batch': batch.name, 'audited_at': s.now(), 'items': [],
              'qwen_calls': 0, 'm4_training_steps_per_case': 250}
    verified_sources = {}
    for item in manifest['items']:
        if not item['complete']:
            continue
        key = item['key']
        m3 = Path(item['states']['m3']['output'])
        m4 = Path(item['states']['m4']['output'])
        summary = read_json(m4 / 'summary.json')
        checks = summary['checks']
        for name in ('object_file_sha256_unchanged', 'object_scene_rows_bitwise_identical',
                     'unselected_background_bitwise_identical', 'background_geometry_bitwise_identical',
                     'non_rgb_background_colors_bitwise_identical', 'original_sources_sha256_unchanged'):
            assert checks[name] is True, (key, name)
        assert checks['training_steps'] == 250
        assert not checks['di_sds_executed']
        assert checks['qwen_calls'] == checks['trellis_calls'] == 0
        assert checks['background_added'] == checks['background_deleted'] == 0
        assert s.sha(item['input_path']) == item['input_sha256']
        assert s.sha(m3 / 'object_world.ply') == summary['candidate_sha256']['object']
        assert s.sha(m4 / 'scene_candidate_step250.ply') == summary['candidate_sha256']['scene']
        assert s.sha(summary['candidate_background']) == summary['candidate_sha256']['background']
        count = vertex_count(m3 / 'object_world.ply')
        assert count == checks['object_count']
        for name in ('scene_final.ply', 'scene_candidate_step250.ply'):
            assert vertex_count(m4 / name) == count + checks['background_count']
        for stage in ('m1', 'materials'):
            rid = item['source_heads'][stage]
            if rid not in verified_sources:
                verified_sources[rid] = v.verify_outputs(v.run(rid))
            assert verified_sources[rid], (key, stage, 'source integrity')
        folder = batch / 'results' / item.get('result_subdir', key)
        videos = {}
        if (folder / 'render_receipt.json').exists():
            receipt = read_json(folder / 'render_receipt.json')
            assert receipt['object_sha256'] == summary['candidate_sha256']['object']
            assert receipt['scene_final_sha256'] == s.sha(m4 / 'scene_final.ply')
            for kind in ('orbit', 'trajectory'):
                videos[kind] = probe(folder / (kind + '_m3_m4.mp4'))
                assert int(videos[kind]['streams'][0]['nb_read_frames']) == receipt[kind + '_frames']
                if (folder / 'candidate_video_receipt.json').exists():
                    videos[kind + '_actual250'] = probe(folder / (kind + '_m3_actual250_final.mp4'))
                    assert int(videos[kind + '_actual250']['streams'][0]['nb_read_frames']) == receipt[kind + '_frames']
                elif not args.allow_incomplete:
                    raise ValueError('Missing actual-candidate video: ' + key)
        elif not args.allow_incomplete:
            raise ValueError('Missing final videos: ' + key)
        runs = s.rows('SELECT external_api,status FROM generation_records WHERE experiment_id=:e',
                      e=item['experiment_id'])
        assert not any(row['external_api'] for row in runs), key
        timings = {name: s.one('SELECT seconds FROM generation_records WHERE run_id=:r', r=rid)['seconds']
                   for name, rid in [('native_m3', item['runs']['m3']), ('reviewed_m3', item['states']['m3']['id']),
                                     ('m4', item['states']['m4']['id'])]}
        timings['failed_native_attempt_seconds'] = [
            (datetime.fromisoformat(r['finished_at']) - datetime.fromisoformat(r['started_at'])).total_seconds()
            for r in (v.run(rid) for rid in item.get('prior_m3_runs', []))
            if r['finished_at'] and r['started_at']]
        if (m3.parent / 'pose_retry.json').exists():
            timings['additional_pose_refit_seconds_approx'] = read_json(m3.parent / 'pose_retry.json')['additional_wall_seconds_approx']
        report['items'].append({
            'key': key, 'description': item['description'], 'object_count': count,
            'background_count': checks['background_count'], 'training_steps': 250,
            'selected_step': summary['selected_step'], 'guard_passed': summary['candidate_guard_passed'],
            'anchor_visibility_before': summary['baseline']['anchor']['visibility'],
            'anchor_visibility_actual250': summary['candidate']['anchor']['visibility'],
            'anchor_outside_union_mae': summary['candidate']['anchor']['outside_union_mae'],
            'checks': checks, 'object_sha256': summary['candidate_sha256']['object'],
            'stage_seconds': timings,
            'videos': videos, 'web_url': item['web_url'], 'm3': str(m3), 'm4': str(m4)})
        print(key, 'audit passed', flush=True)
    report['audited_cases'] = len(report['items'])
    report['candidate_guard_passed_cases'] = sum(x['guard_passed'] for x in report['items'])
    report['nonzero_checkpoint_selected_cases'] = sum(x['selected_step'] > 0 for x in report['items'])
    report['input_background_retained_cases'] = sum(x['selected_step'] == 0 for x in report['items'])
    report['source_runs_unchanged'] = verified_sources
    report['resource_retries'] = []
    for path in batch.glob('case*_resource_retry.json'):
        retry = read_json(path)
        if retry.get('raw_sha256'):
            assert s.sha(retry['reused_raw_checkpoint']) == retry['raw_sha256']
        report['resource_retries'].append(retry)
    report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=s.ROOT, text=True).strip()
    code = [s.BASE / 'modules' / n for n in ('m3_trellis_generation.py', 'm4_refinement.py')]
    code += [s.BASE / 'core' / n for n in ('alignment.py', 'mesh_to_gaussian.py',
                                        'multiview_masks.py', 'local_background_optimizer.py', 'visibility_loss.py')]
    code += [s.BASE / 'scripts' / n for n in (
        'complete_qwen_results_batch.py', 'review_qwen_sam_candidates.py', 'refine_reviewed_sam.py',
        'extract_qwen_rembg.py', 'refit_m3_reference.py', 'postprocess_qwen_batch.py',
        'parallel_qwen_native.py', 'collect_qwen_batch_results.py', 'render_qwen_batch_final.py', 'audit_qwen_batch.py')]
    report['code_sha256'] = {str(p.relative_to(s.ROOT)): s.sha(p) for p in code if p.is_file()}
    write_json(batch / 'audit_report.json', report)
    if args.allow_incomplete:
        return
    text = '''9 USER-GENERATED QWEN IMAGES: M3 + M4 RESULTS

Each case has an independently generated TRELLIS.2 asset, 200000 Gaussian samples,
an RGB + silhouette pose check, and a real 250-step local-background M4 run.
The new object's parameters stay frozen during M4. No Qwen or DI-SDS calls.

IMPORTANT: Actual training candidate and guarded final are different outputs.
scene_candidate_step250.ply is the actual optimized candidate. scene_final.ply
uses the checkpoint accepted by the existing environment guard. selected_step=0
means the candidate was rejected and the original M1 background was retained.
Do not describe rejected candidates as successfully preserved environments.

Foreground adaptation: cases03B/05B use installed RMBG-2.0 because SAM omitted
object parts or merged adjacent equipment. All other cases use reviewed SAM.
Only alpha changes; input RGB files are preserved and SHA256 checked.
Box-like devices required RGB-assisted yaw selection after native silhouette fit.
Native assets and initial runs remain intact. Metallic/roughness PBR shading is
not represented by the current base-color SH0 Gaussian conversion.

SERVER ROOT: /mnt/sda/ubuntu/lkx/papers/paper_eli
BATCH: industrial_swap_v2/outputs/qwen9_completion_20261107
WEB: http://127.0.0.1:18086/static/batches/qwen9_completion_20261107/index.html

manifest.json records original M3 jobs and input provenance.
final_manifest.json records the reviewed M3 and M4 stage heads.
aligned/<case>/config.yaml and logs contain exact effective configurations.
aligned/<case>/m3: object_world.ply, scene_coarse.ply, GLB, pose evidence.
aligned/<case>/m4: candidate and guarded PLYs, reports, masks and validation views.
results/<case>: links to final PLYs/GLB and same-camera renders/videos.

This portable bundle contains previews, reports and reproduction scripts.
Large PLY/GLB assets remain on the server and are downloadable from the Web page.
The original M1/M2 preparations and historical experiments were not overwritten.

SAFE READ-ONLY RE-EXPORT (run from SERVER ROOT):
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.scripts.collect_qwen_batch_results --batch BATCH
CUDA_VISIBLE_DEVICES=3 envs/gsle/bin/python -m industrial_swap_v2.scripts.render_qwen_batch_final --batch BATCH
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.scripts.audit_qwen_batch --batch BATCH
Replace BATCH with the relative directory above. These commands do not train.
To repeat training, create a separate experiment and use its copied config.
Do not rerun training against a completed result directory.
'''
    (batch / 'RESULTS_README.txt').write_text(text, encoding='utf-8')
    chinese = [f'9 个工业设备替换结果：{report["audited_cases"]}/9 已实际运行 M3 与 M4。',
        'M4 每组 250 步，固定全部新设备高斯，仅优化局部背景。没有调用 Qwen 或 DI-SDS。',
        f'末轮候选通过现有环境保护检查：{report["candidate_guard_passed_cases"]}/9。',
        f'最终选用较早优化检查点：{report["nonzero_checkpoint_selected_cases"]} 组；保留原始输入背景：{report["input_background_retained_cases"]} 组。',
        'scene_candidate_step250.ply 是实际优化结果；scene_final.ply 使用保护规则选择的背景。',
        'selected_step=0 表示退回原始 M1 背景，因此不能把该文件描述为采用了 250 步优化。',
        '视频三列依次为 M3、实际 250 步候选、保护规则选出的场景。', '',
        '编号 | Anchor 设备可见贡献：M3 → 实际250步 | Mask外变化MAE | 最终选择步数']
    for entry in report['items']:
        chinese.append(f'{entry["key"]} | {entry["anchor_visibility_before"]:.2%} → {entry["anchor_visibility_actual250"]:.2%}'
                       f' | {entry["anchor_outside_union_mae"]:.5f} | {entry["selected_step"]}')
    chinese += ['', '质量限制与适配：',
        '03B、05B 的 SAM 分割不完整，使用已安装的 RMBG-2.0 提取前景，输入 RGB 保持不变。',
        '部分案例增加独立朝向搜索或经检查的原生参考正面约束；原对齐记录保留，详见各例配置和记录。',
        '07A、10C 的 16GB 显存上限不足，预算提高到32GB；10C 从已保存检查点继续导出。',
        '金属反光、玻璃与小结构在当前 Mesh-to-Gaussian 基色转换中明显弱化；原始 GLB 单独保留。',
        '设备的小结构和材质仍存在偏差；M4 不修改设备几何。各例对齐检查见 final_manifest.json。',
        '原场景重建残影、未被新设备覆盖的旧区域仍可能存在。', '',
        '检查与下载：http://127.0.0.1:18086/static/batches/qwen9_completion_20261107/index.html',
        '服务器目录：' + str(batch),
        '本压缩包包含预览、视频、配置、日志与复现脚本。完整 PLY / GLB 保存在服务器，可从检查页下载。']
    (batch / 'RESULTS_ZH.txt').write_text('\n'.join(chinese) + '\n', encoding='utf-8')
    bundle = batch / (batch.name + '_previews.zip')
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in ('audit_report.json', 'manifest.json', 'final_manifest.json', 'postprocess.json', 'adaptations.json', 'RESULTS_README.txt', 'RESULTS_ZH.txt'):
            archive.write(batch / name, name)
        for path in batch.glob('case*_resource_retry.json'):
            archive.write(path, path.name)
        for path in (batch / 'review').glob('*overview.jpg'):
            archive.write(path, 'overview/' + path.name)
        for item in manifest['items']:
            key = item['key']; m3 = Path(item['states']['m3']['output']); m4 = Path(item['states']['m4']['output'])
            native = v.run(item['runs']['m3'])
            native_output = Path(s.unpack(native['outputs'])['directory'])
            archive.write(native['config_path'], key + '/native_m3_config.yaml')
            archive.write(native['log_path'], key + '/native_m3.log')
            for name in ('trellis_generation.json', 'resource_policy.json', 'object_local.json', 'trellis_asset_preview.png'):
                archive.write(native_output / name, key + '/native_m3/' + name)
            archive.write(Path(item['sam_review_directory']) / 'selection.json', key + '/foreground_selection.json')
            if (m3.parent / 'pose_retry.json').exists():
                archive.write(m3.parent / 'pose_retry.json', key + '/pose_retry.json')
            archive.write(item['input_path'], key + '/qwen_input.png')
            for name in ('stages_same_camera.jpg', 'orbit_m3_actual250_final.mp4', 'trajectory_m3_actual250_final.mp4', 'm4_report.json'):
                archive.write(batch / 'results' / item.get('result_subdir', key) / name, key + '/' + name)
            for source, target in ((m3 / 'alignment_comparison.jpg', 'alignment.jpg'),
                                   (m4 / 'validation/anchor_comparison.png', 'actual250_comparison.png'),
                                   (m4 / 'validation/supervision_union_masks.png', 'dual_masks.png'),
                                   (m4.parent / 'config.yaml', 'config.yaml'),
                                   (m4.parent / 'm3.log', 'm3_pose.log'),
                                   (m4.parent / 'm4.log', 'm4.log')):
                archive.write(source, key + '/' + target)
        for relative in report['code_sha256']:
            if '/scripts/' in relative:
                archive.write(s.ROOT / relative, relative)
    report['preview_bundle'] = {'path': str(bundle), 'bytes': bundle.stat().st_size, 'sha256': s.sha(bundle)}
    write_json(batch / 'audit_report.json', report)
    print(json.dumps({'audited': report['audited_cases'], 'guard_passed': report['candidate_guard_passed_cases'],
                      'bundle': report['preview_bundle']}, indent=2), flush=True)


if __name__ == '__main__':
    main()

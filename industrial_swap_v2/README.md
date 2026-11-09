# Industrial Swap v2：正式四模块流程

**LangSplat 定位 → Qwen 深度参考二维生成 → PBR 多视角设备 Gaussian 训练 → 双 Mask 场景融合与背景优化。**

实验仅在 `ubuntuA6000:/mnt/sda/ubuntu/lkx/papers/paper_eli` 执行。M1/M2 保持原样。新版 M3 使用 TRELLIS.2 GLB 与工厂 HDRI 的 RGB/Depth 监督；空间对齐移入 M4。M4 保留 `old_new_union_v2` 的优化权限和损失，不加载扩散教师。旧配置缺少 `m3.method` 时仍可读取与复用历史结果。

## 代码与数据接口

| 模块 | 正式入口 / 核心代码 | 输入与输出 |
|---|---|---|
| M1 | `modules/m1_localization.py` | 原场景、LangSplat 索引 → `target_gaussians.ply`、`background_gaussians.ply`、索引与 `manifest.json` |
| M2 | `modules/m2_image_generation.py`；`core/cameras.py`、`depth.py`、`qwen_client.py`、`segmentation.py` | M1 空间、AVP → RGB/Depth/Mask 条件、Qwen 图像、SAM 前景和 Anchor 相机 |
| M3 | `modules/m3_trellis_generation.py`；`core/pbr_pipeline.py`、`pbr_gaussian_optimizer.py`、`mesh_to_gaussian.py`；`scripts/blender_pbr_views.py` | PBR GLB + HDRI → 36 视角 RGB/Depth/Mask/Camera → 初始高斯 → `object_refined_local.ply`、最佳 checkpoint、独立验证渲染 |
| M4 | `modules/m4_refinement.py`；`core/scene_integration.py`、`multiview_masks.py`、`local_background_optimizer.py`、`m4_outputs.py` | 局部设备 → 空间对齐（含 SH 旋转）、`object_world.ply`/`scene_initial.ply` → 双 Mask/遮挡证据 → 局部背景优化与最终场景 |

双 Mask 配置仅属于 `m4.multiview_supervision`，构建发生在 M4 优化前。旧配置的 `m3.visibility_initialization` 和旧 M3 缓存仅为只读兼容保留，不再由 M3 生成。

共享入口为 `run.py`；`core/workflow.py` 负责正式流程校验与新实验配置迁移。Web 的 `adapter.py` 调用相同模块，素材库继续支持图像、Mesh、局部/世界 Gaussian 复用。无需为复用素材调用 Qwen 或 TRELLIS。

## M4 监督与权限

M4 默认 `integration.alignment: rasterized_reference`。`core/alignment.py` 的
`fit_reference` 调用 `core/gaussian_registration.py`：在保存的 M2 Anchor 内外参下
渲染完整 Gaussian 的 RGB、Alpha 和相机 Z 深度，搜索 24 个 Yaw 与三种尺寸初值，
对不同朝向候选进行可微配准。只优化统一尺度、Yaw 和两个水平位移，设备本身不训练。
损失为软轮廓 IoU、轮廓距离、低权重归一化 RGB 结构/色度，以及三维占地中心和越界约束。
RGB 不拟合曝光或重新着色。三维初值使用 M1 实测占地中心和 M2 的精确生成框；
支撑高度固定为 M2 保存的底面，不从 Anchor 相机朝向重新定义设备坐标。

导出时对位置、尺度、协方差旋转和 SH0–SH3 使用同一空间变换，并验证导出渲染与
可微配准渲染的一致性。`alignment/` 保留候选、指标、RGB/Alpha/Depth、参考 Mask
叠加和新旧同相机检查。轮廓或三维越界检查不合格时停止，不继续优化背景。
`verified_transform` 与历史 `anchor_silhouette` 仅供显式选择的历史配置兼容。

复用已完成案例重新配准（在服务器项目根目录运行）：

```bash
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.scripts.prepare_reference_alignment \
  --source-config industrial_swap_v2/outputs/pbr_10cases_m4_2000_20261108/case_02_B/config.yaml \
  --destination outputs/case02_raster_alignment_v1 --gpu 3 --iterations 2000
CUDA_VISIBLE_DEVICES=3 SWAP_GPU=3 envs/gsle/bin/python -m industrial_swap_v2.modules.m4_refinement \
  --config industrial_swap_v2/outputs/case02_raster_alignment_v1/config.yaml
CUDA_VISIBLE_DEVICES=3 SWAP_GPU=3 envs/gsle/bin/python -m industrial_swap_v2.scripts.inspect_reference_alignment \
  --config industrial_swap_v2/outputs/case02_raster_alignment_v1/config.yaml
```

准备脚本拒绝覆盖已有目录。新的位姿必须重新构建 M4 Mask 缓存；旧案例的定位、
Qwen 图像、独立 PBR Gaussian 和背景文件保持只读。对称设备的单视图姿态仍可能不唯一，
候选分数及生成框越界量都随结果保存，不把视觉相似性解释为准确的物理测量。

若参考图拟合的最优候选超过既有生成空间限制，使用相同损失进行一次约束内
精配准：每步仅将尺度和水平平移投影回可行范围，不调整支撑高度、不修改设备，
也不放宽原有 IoU/越界阈值。`alignment/constrained_refinement.json` 保存这一步的
输入候选及结果。批量入口为 `scripts/run_registration_batch.py`，各案例独立输出；
已验证的 case02 可直接复用，M1–M3 均不重新运行。

经人工确认的高度适配可通过新实验的 `m4.integration.space_override` 指向
`m4_height_adaptation.json`。此文件保存原 M1/M2 的 SHA256 和新旧生成框；
入口强制校验底面、方向、水平占地不变，且新高度不超过原 M1 编辑空间顶面。
它不修改 M2，也不会在配准失败后自动启用。本批次仅 case05 获准使用，
相关文件和调整说明随该案例结果打包保存。

每个相机分别渲染旧目标的可见贡献以及固定新设备的独立 Alpha。环境区域严格为 `M_env = ~(M_old | M_new)`；真实照片损失另保留既有的小幅边界保护、图像边缘和局部支持限制。大包围盒仅用于合法前景的保守深度判定，不再作为环境 RGB 的主要排除区域。

M4 的 `multiview_supervision/` 保存 `cache/*.npz`、Mask/Alpha/Depth、相机清单、`rgb_indices.npy`、`occlusion_indices.npy` 和多视角遮挡证据。Anchor 的 M2 Mask 仅用于一致性审计，其他视角使用各自独立渲染。遮挡证据仍使用真实渲染贡献梯度、三维协方差支持、视角数量/角度和合法前景保护。

M4 固定新设备全部参数、全部高斯几何与数量。RGB 集合允许更新局部背景 SH0 颜色和透明度；遮挡集合只通过可见性及保持损失更新透明度，两组梯度权限独立。未选中背景逐字节冻结。默认 250 步，可设置 200–10000 步；优化结束即导出，没有后续训练阶段、地面补全或删除操作。

本次清理保持上一版 union 路径的损失、有效默认值、采样顺序、梯度权限、训练循环及检查点选择行为。历史 union 缓存中的无效旧参数在兼容性比较时忽略；有数值影响的参数、源文件校验值变化均不可复用。历史 bbox 缓存不会进入新 M4；复用旧 M3 时，必要的新 Mask 写入本次 M4 目录，避免修改原资产。

## M4 输出约定

- `m4/background_optimization/background_optimized.ply`：通过既有环境保持约束的选中背景。
- `m4/scene_final.ply`：选中背景与对齐后的设备合成；`object_fixed.ply` 与 `m4/object_world.ply` SHA256 相同（历史流程对应 `m3/object_world.ply`）。
- `m4/background_optimization/background_candidate_stepN.ply`、`m4/scene_candidate_stepN.ply`：最后一步的真实候选，可能未通过检查。
- `m4/background_optimization/`：训练 JSONL、检查点、基线/最终指标、两组索引。
- `m4/validation/`：同相机候选对比、逐视角指标、双 Mask 可视化；`m4/summary.json` 汇总选择状态。
- `m4/renders/anchor_after.png` 始终对应 `scene_final.ply`；末步候选另存 `anchor_candidate_stepN.png`。

保留已有 `scene_visibility_optimized.ply` 缓存文件名以兼容先前 union 结果，它不是额外阶段。若所有检查点都未通过保持约束，选中结果退回 step 0，候选仍单独公开。**完成短程执行不等于候选通过质量检查。** 既有 2026-11-06 250 步实验即为这种情况；本次没有重新训练或改写其结果。

## 命令行和素材复用

默认配置为 `configs/default.yaml`，新的默认输出位于 `outputs/official`，记录/日志位于 `records/official`、`logs/official`，避开历史 `outputs/m1`–`m4`。每个新实验请使用独立目录；输入或有效配置变化时不要覆盖完成结果。

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
envs/gsle/bin/python industrial_swap_v2/run.py --config NEW_CONFIG.yaml --stage m1
envs/gsle/bin/python industrial_swap_v2/run.py --config NEW_CONFIG.yaml --stage m2 --prepare-only
# 生成阶段仅在明确需要时执行；已有图片/模型可从 Web 素材库直接复用。
envs/gsle/bin/python industrial_swap_v2/run.py --config NEW_CONFIG.yaml --stage m3
envs/gsle/bin/python industrial_swap_v2/run.py --config NEW_CONFIG.yaml --stage m4
```

针对已有完整 M1/M2/M3 建立新 M4 实验，保留兼容的独立入口：

```bash
envs/gsle/bin/python -m industrial_swap_v2.scripts.run_background_only \
  --source-config EXISTING_CONFIG.yaml --destination outputs/NEW_EXPERIMENT \
  --steps 250 --gpu AVAILABLE_GPU --stage all
```

该命令复用源 M1/M2、复制现有 M3 设备及变换，明确生成新配置，不执行生成网络。M4 中断时保存现场，请另建实验；已完成优化但导出中断可读回已完成背景继续导出。清理验证期间没有执行上述训练命令。

Web 操作详见 `webapp/README.md`。复制历史实验保留素材选择。新阶段缓存版本为 `web-v8-pbr-local-gaussian` / `web-v8-pbr-scene-integration`，原始 TRELLIS GLB 生成缓存仍可复用；PBR M3 素材输入需带材质 GLB。M4 Mask 设置不影响 M3 训练缓存。历史结果与文件下载保持可读。

## PBR 训练与 case00 验证

配置集中在 `m3.pbr`（Blender、HDRI、光照、渲染和训练）以及 `m4.integration`（对齐）。
默认 36 视角：12 个方位角 × -10/10/30 度俯仰，512²、Cycles 64 samples；每层 150/330 度共 6 视角只用于验证。
`RGB Loss` 是白底 display-sRGB L1；`Depth Loss` 是物体有效像素的相机 Z L1，除以 GLB 包围盒对角线。仅这两项损失，权重 1 / 0.2；不含 Mask Loss。
初始化保留 GLB Y-up 坐标、面积加权表面采样、法线方向、纹理 SH0、面积采样间距尺度；训练调整每个高斯的 SH3/Opacity/尺度/有界位置偏移，旋转与数量固定。
Blender 导出的浮点 EXR 必须先经过 `validate_pbr_calibration.py` 的独立三角面求交验证；4.5.4 的实测定义为 camera Z，不能误用欧氏射线距离。
SH 在空间旋转时按实球谐基逐阶变换，保持同一局部观察方向的颜色，不丢弃训练出的高阶系数。

case00 使用既有 GLB 和已验证的相同 GLB 变换以保持位置可比；其他案例默认自动 Anchor 轮廓对齐。新设备在 M4 始终固定，仅优化局部背景。
本轮独立输出：`outputs/pbr_case00_20261108`，复现入口：

```bash
# 只对新目录运行 prepare；不要覆盖完成实验。
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.scripts.prepare_pbr_case00 --output outputs/NEW_PBR_CASE00 --gpu 2
envs/gsle/bin/python industrial_swap_v2/run.py --config industrial_swap_v2/outputs/NEW_PBR_CASE00/config.yaml --stage m3
envs/gsle/bin/python industrial_swap_v2/run.py --config industrial_swap_v2/outputs/NEW_PBR_CASE00/config.yaml --stage m4
```

依赖：Blender 4.5.4 LTS（项目 tools 下官方二进制、下载 SHA256 校验），项目 `envs/gsle` 中 trimesh 4.6.8、rtree 1.4.1、OpenEXR 3.3.2。
项目依赖安装入口为 `bash industrial_swap_v2/scripts/setup_pbr.sh`，通过既有本机代理隧道下载，不修改系统环境。
Blender 保留 GLB 原材质/纹理，Standard/sRGB 输出，HDRI 强度 1、旋转 0；照明不会把室内照片重投影成纹理。
学到的是固定 HDRI 下的 SH3 外观近似，不能等同于可任意重新打光的 PBR 材质。M4 也不会补全真实照片中未观测的地面。

## M2 自适应空间与可见性编辑区域（2026-11-07）

`core/m2_space.py` 从实际 M1 目标高斯的中心、协方差和透明度估计稳健尺寸，在 M1 场景竖直方向建立局部坐标。支撑位置默认取观测底部；只有与已有 M1 地面平面足够接近时才吸附到该平面，避免将架高设备强行落到地面。

英文描述的设备主体匹配结构先验（机械臂、SCARA、加工中心、柜体、龙门工位、输送机、圆筒设备、推车/工作台）。这些是显式比例启发式，不是语言模型预测的真实物理尺寸；未识别类型保守沿用测量尺寸。高度变化有上限，参数和命中的短语写入 `m2/generation_region.json`。不再使用“高度统一等于最长边”的规则。

`core/m2_edit_mask.py` 在 Anchor 上独立渲染目标 Alpha，并通过原场景的目标贡献及深度检查得到可见轮廓。按投影尺度去除小孤立区域、闭合窄裂缝、填补有限面积孔洞；沿新旧三维边长差对应的投影方向扩展轮廓，扩展限制在新生成框内。保留原目标清除轮廓，因此二维编辑区域可以包含新生成框之外的小部分旧设备。

前景保护使用真实高斯合成权重：33 个深度阈值下渲染背景的累积贡献，再查询生成体前方/内部的遮挡质量。它避免把后方多层背景导致的较大深度方差误认为禁止扩展。阈值之间采用线性插值，深度以高斯中心相机 z 为准；不是精确表面深度。高置信前景和邻近占据被保留，证据不足不授权扩展。扩展大部分被阻挡或旧设备贡献保留不足时，报告要求人工检查。

- `generation_region.json`：独立三维生成框、精确局部坐标、尺寸来源、支撑关系、语义先验，供 M3 尺度拟合使用。
- `edit_mask.png`：可见旧轮廓与受约束扩展的并集，不是整框投影。
- `position_hint.png`：仅编辑 Mask 内叠加紫色位置提示；前景孔洞及 Mask 外 RGB 不变。
- `depth_background.png`：仅编辑 Mask 内设为未知；Mask 外深度与 `depth.png` 逐字节相同。
- `old_visible_mask.png`、`old_clean_mask.png`、`generation_bbox_mask.png`、`extension_proposal.png`、`protected_foreground_mask.png`、`mask_report.json`、`spatial_depth.npz`：检查依据。

M3 的生成和素材导入均优先读取 M2 的精确 `world_transform`，并验证生成空间 SHA256；历史 M2 仍使用其原 M1 坐标。M2 条件缓存包含算法版本、设备描述、有效参数及 M1 文件哈希；不同设备描述不再共享 Mask/生成框。完成版本不可原地切换算法或描述，须新建 M2 准备版本。历史 30 组 Qwen 输入不会自动更新或覆盖。

只准备输入：`--stage m2 --prepare-only`。同相机实际渲染检查入口为 `python -m industrial_swap_v2.scripts.validate_m2_spatial_conditions --batch <已有批次 manifest.json> --output <新的独立目录>`，默认检查三个位置的五种描述；入口禁止 Qwen，不运行 M1、SAM、TRELLIS、M3/M4 或训练。主 AVP 仍使用已有亮度/可见性评分，对新的三维空间选择相机。

## 历史代码与依赖

`legacy/pre_four_module_cleanup/` 保存旧 DI-SDS、地面补丁、调整高度诊断、OAGOF 删除、旧 bbox/多视角优化及相关脚本/文档；`provenance.json` 记录源路径和 SHA256。正式入口不导入归档代码。完整可运行历史基线为提交 `071e179d1d63fed4605d1108f17cadfe293b5647`，比较实验请在独立 checkout 和新输出目录运行。

原 `gsle/` 未修改。正式流程仍只读复用其中相机/高斯/raster/SH 工具及 AVP/SAM 等已有适配依赖；现有环境可能仍安装扩散相关包，但正式 M4 不导入或初始化它们。Qwen、SAM、TRELLIS.2 的版本、模型缓存、官方 revision 和生成约束维持原样。Qwen 深度是视觉参考，不是 ControlNet 条件。当前 SAM 适配的既有限制及上传 RGBA 路径见 Web 文档。

## 轻量验证

CPU 测试覆盖 Mask 定义、两组梯度权限、固定设备、场景合成、历史配置迁移/缓存、图像/模型素材复用和 Web 接口。关键函数及完整训练循环与归档版本进行 AST 等价回归。只读已有结果检查另外核对 41 个相机缓存及 Gaussian 保持信息，不执行渲染、训练或网络生成。具体结果见 `CLEANUP_REPORT.md`。

## 2,000 步与分段保存

按后续运行要求，M4 允许 200–10000 步（默认仍为 250）；`m4.save_every: 500` 每 500 步及最终一步保存 `m4/snapshots/iteration_000500` 等独立目录。包含实际迭代的背景 PLY、合成场景 PLY、优化器/采样状态检查点、41 视角指标和 4 个相机预览。快照反映当时训练状态，不以保持约束筛选结果替代；`snapshot.json` 单独记录是否通过约束及文件 SHA256。独立入口支持 `--steps 2000 --save-every 500`，新实验从原始 M1 背景开始，复用固定设备与通过校验的监督缓存。


M4 支持从实际迭代快照连续训练：使用 `--source-config <旧实验/config.yaml> --destination <新目录> --resume-checkpoint <旧实验/m4/snapshots/iteration_002000/checkpoint.pth> --steps 10000 --save-every 2000`。总步数包含已完成的 2000 步；新运行执行 2001–10000，保存 4000/6000/8000/10000。恢复背景参数、Adam 状态、相机采样顺序及随机状态，并验证来源哈希、监督签名和冻结缓冲区。除总步数与保存间隔外不允许改变优化参数；原实验保持只读。新运行 `background_optimization/resume_verification.json` 记录恢复检查。默认 `scene_final.ply` 仍经过环境保持检查；实际迭代结果位于 snapshots。

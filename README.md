# Industrial 3DGS editing reproduction

## GSLE 三维物体替换（2026-11-05）

新增 [GSLE 代码与运行说明](gsle/README.txt)，基于同一场景自动定位整台黑色组件及白色底座，并生成黄色工业机械臂。已实际完成 LangSplat 三尺度训练、AVP、深度 Inpainting、SAM、6500 步 Stable Zero123 3D-SDS、合成，以及两种无条件分支解释各 2000 步 DI-SDS。此实现依据论文及补充材料编写，复用官方组件，并非作者官方 GSLE 源码。

**流程已完整运行，但高质量替换目标尚未达成。** 保留背景条件的 `text_only` 版本正面可以辨认机械臂，侧后方仍存在块状结构、孔洞和漂浮点；`drop_all_eq7` 版本产生黄色雾状伪影，作为负结果保留。两套实际模型、同相机对照、350 视角原轨迹渲染和 360° 视频均已导出并核验。原速视频为 175 秒，29.17 秒版本是 6 倍速预览。

- [实际运行报告与局限](gsle/REPORT.txt)
- [论文算法、歧义与适配记录](gsle/PAPER_NOTES.txt)
- [模块输入输出与坐标约定](gsle/interfaces.json)
- [交付索引及逐文件 SHA256](records/gsle/deliverables.json)
- [最终视觉检查](records/gsle/final_visual_review.json)与[两种条件分支对照](records/gsle/cfg_comparison.json)

服务器较好版本位于 `outputs/gsle/ablations/text_only_cfg`，完整结果归档为 `gsle_results.tar.gz`。本机执行 `powershell -ExecutionPolicy Bypass -File gsle/download_results.ps1` 可下载模型及预览到 `results/gsle`，并自动校验 SHA256。大型模型、图片、视频和权重不进入 Git。

## 原有四方法颜色编辑基线

在同一段工业场景视频上，依次运行 GaussianEditor、DGE、DreamCatalyst 和 EditSplat 的官方开源代码。仓库包含数据预处理、环境安装、模型下载、场景适配、运行、渲染与评价脚本，以及本次实验的版本记录和实际测量结果。

**当前结果：四种方法均完成实际运行，但均未明显将指定设备外壳改红。流程完成不代表编辑目标达成。** 本仓库记录的是可复现的负结果基线，不能作为四篇论文在其他场景上的性能结论。

## 实验目标与数据

- 输入：`20261016-142321.mp4`，174.89655 秒，2560 × 1440，29 fps。
- 输入 SHA256：`061413776d7e163c53aa37cd556018bd65356df2128f4929a55ed90645c67116`。
- 统一指令：**将画面中央设备的外壳改成红色，保持其他物体不变。**
- 执行英文：`Change the housing of the industrial machine in the center to red, keeping all other objects unchanged.`
- 固定目标：原视频约 75 秒处，黑色组件下方的白色设备底座／外壳。三维目标和人工检查记录见 `records/target_geometry.json` 等文件。
- 以 2 fps 抽取 350 帧，COLMAP 注册 350/350 个视角；基础 3DGS 训练 30,000 步，306 个训练视角、44 个留出测试视角。
- 四种方法使用同一基础模型、相机划分和主编辑指令，保留各自官方的采样及优化策略。

展示视频使用 350 个相机视角、12 fps，时长约 29.17 秒，即覆盖约 175 秒原视频轨迹的 6 倍速展示。原视频 75 秒对应展示视频约 12.5 秒。

## 实际结果

下表的图像指标来自目标可见的 24 个测试视角。耗时包含方法加载、分割、编辑、优化和自带渲染，不包含共同预处理、基础模型训练、安装下载、失败重试及额外统一渲染。服务器为共享环境。

| 方法 | 耗时（分钟） | 目标红色比例 | 背景 PSNR（dB） | 背景 LPIPS |
|---|---:|---:|---:|---:|
| GaussianEditor | 16.50 | 0.018% | 30.981 | 0.0844 |
| DGE | 6.07 | 0.004% | 31.630 | 0.0743 |
| DreamCatalyst | 13.60 | 0.354% | 37.381 | 0.0236 |
| EditSplat | 70.31 | 0.156% | 27.900 | 0.1170 |

目标在实际 RGB 中仍主要为白色，部分方法出现偏色或边缘伪影。背景保持分数较好、重投影误差较低，均不能单独证明编辑成功。完整说明见 [REPORT.txt](REPORT.txt)，逐视角指标见 [records/common_metrics.json](records/common_metrics.json)。

## 运行

详细环境安装、模型下载、预处理和补丁应用顺序见 [README_reproduction.txt](README_reproduction.txt)。脚本当前面向已有服务器环境，工作目录为 `/mnt/sda/ubuntu/lkx/papers/paper_eli`；迁移服务器时需要调整脚本中的目录、CUDA、Conda 和代理配置。

在已准备好数据、模型和独立环境的服务器上重新运行：

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
source scripts/common.sh
bash scripts/replay_baseline.sh repeat1
```

`repeat1` 必须是尚未使用的标签。脚本按顺序运行四种方法、检查 PLY 并进行同相机渲染，不覆盖原输出。新结果位于 `outputs/<method>_repeat1`，指标位于 `records/replay_repeat1_metrics.json`。运行后仍需人工查看实际 RGB。

下载和安装使用本机 HTTP 代理经 SSH 转发到服务器的 `127.0.0.1:17891`。已核验缓存的重跑启用 Hugging Face 离线模式。仓库不包含 SSH 密钥、输入视频、预训练权重或运行环境。

## 代码与记录

- `scripts/`：下载、安装、预处理、官方方法启动、兼容性及实例选择补丁、统一渲染、评价与打包脚本。
- `records/reproducibility/`：实际运行时的官方源码差异、额外源文件、上游提交和脚本校验值。
- `records/*.pip.freeze.txt`：各独立环境的实际包版本。
- `records/runs/*/run.json`：实际命令、运行状态、耗时、显存采样峰值及失败尝试记录。
- `records/final_comparison.json`：结果、模型路径、SHA256 和质量检查汇总。
- `records/official_checkpoints.json`：官方 checkpoint 的位置和校验值。
- `records/final_artifact_verification.json`：模型、渲染数量及视频帧数核验。

`records/` 中的结果是历史实验快照，不代表新机器已完成安装。运行就绪标记、完整日志、图像、视频、模型和压缩包不进入 Git。完整交付包 `reproduction_results.tar.gz` 和大型 PLY/checkpoint 保留在原工作目录；压缩包校验值为仓库中的 `reproduction_results.tar.gz.sha256`。

实际补丁用于旧依赖兼容、DGE 数值域修正、同一物理设备的候选 mask 选择和诊断输出。未用手工染色替代方法输出。应用补丁请使用复现说明中的幂等脚本，不要再重复叠加保存的 diff。

`scripts/15_review_and_evaluate.sh` 包含本次人工检查后的固定结论，仅为本次运行记录；新实验必须先实际检查图像，再记录对应结论。

## 固定上游版本

| 官方仓库 | 提交 |
|---|---|
| [GaussianEditor](https://github.com/buaacyw/GaussianEditor) | `9249f847c57036266a0bfb3a210681f94264ede9` |
| [DGE](https://github.com/silent-chen/DGE) | `7eed0bc57824707c53b935bd19ea994f19f41eae` |
| [DreamCatalyst](https://github.com/kaist-cvml/DreamCatalyst) | `072cb248ce847cef0e443d70baef7db8cee89ee7` |
| [EditSplat](https://github.com/kuai-lab/cvpr25_EditSplat) | `631f26679f9d66c7099413be441d346185c6442f` |
| [原版 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | `2eee0e26d2d5fd00ec462df47752223952f6bf4e` |

第三方源码由脚本按上述提交获取，不把整个上游仓库复制进本仓库。各上游代码、补丁涉及的原始代码及模型仍受各自许可证和使用条款约束。

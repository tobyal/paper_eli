# Industrial device replacement v2

独立路线：**复用 LangSplat → Qwen 场景编辑 + SAM → 官方 TRELLIS.2 → Mesh 表面高斯化与场景合成 → text_only DI-SDS**。

所有实验在 `ubuntuA6000:/mnt/sda/ubuntu/lkx/papers/paper_eli` 运行。本机仅存代码、输入和下载预览。旧 `gsle/`、脚本及历史结果均保留。

## 本次实际运行结果（2026-11-05）

M1–M4 均已完成。M2 成功生成 1 张 Qwen 图像并完成自动 SAM 提取；M3 使用官方 TRELLIS.2 生成原生 Mesh、带材质 GLB 和 200,000 个新高斯；M4 实际执行 2,000 次 DI-SDS 更新，训练约 32.7 分钟，Torch 显存峰值约 8.69GiB。最终场景共 1,997,425 个高斯。几何字段及保留背景通过逐字节保持校验。

最终场景：`outputs/m4/scene_final.ply`；独立设备：`outputs/m4/object_refined.ply`；原始带材质资产：`outputs/m3/trellis_asset.glb`。前后对比、72 视角环绕和 350 帧轨迹视频位于 `outputs/m4/renders/`。本机已下载主要结果，服务器保留完整中间数据和日志。

服务器和本机均通过 28 项登记产物的大小/SHA256 检查。4 个离线测试通过，涵盖完整 Key 解析、禁止重复 POST、失败后锁定、材质/协方差转换及背景不变性。再次执行 `--stage all` 已验证会复用全部完成阶段。

实际可见的限制：旧设备移除后露出的地面在纯背景渲染中已有拖影和缺失，最终合成仍保留这些问题；生成设备表面也有粗糙细节。本次完成的是所要求的四模块运行基线，尚未解决背景补全或 CAD 级几何规整。可查看 `outputs/m4/renders/anchor_background.png` 区分背景问题与新设备。

## 运行

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
source scripts/common.sh
# 根据 nvidia-smi 的实际空闲情况选择；不能停止其他用户进程。
export SWAP_GPU=2
envs/gsle/bin/python industrial_swap_v2/run.py --stage m1
envs/gsle/bin/python industrial_swap_v2/run.py --stage m2 --prepare-only
envs/gsle/bin/python industrial_swap_v2/run.py --stage m2
envs/gsle/bin/python industrial_swap_v2/run.py --stage m3
envs/gsle/bin/python industrial_swap_v2/run.py --stage m4
# 或使用上述已保存结果继续全部阶段
bash industrial_swap_v2/scripts/run_all.sh
```

统一配置为 `configs/default.yaml`。`project.output_dir` 相对于 **industrial_swap_v2/**；历史输入相对于原项目根。模块写入必须留在新目录。解释器路径保留 virtualenv 的符号链接，不能 resolve 到基础 Python。

## 模块接口

| 模块 | 读取 | 保存至 |
|---|---|---|
| M1 | 原场景 PLY、已有 target/edit_region、target/keep indices、AVP 相机 | `outputs/m1/manifest.json`、旧目标/背景 PLY、原始索引、区域 JSON |
| M2 | M1 区域、原场景、已有 AVP | `outputs/m2/anchor.png`、`edit_mask.png`、完整/背景深度、Qwen 编辑图、SAM RGBA/Mask/白底图、相机、请求与分割记录 |
| M3 | M2 RGBA、相机/Mask、M1 世界区域 | `outputs/m3/trellis_native.pt`、带 PBR 的 GLB、官方资产预览、局部/世界物体 PLY、粗合成 PLY、变换矩阵、视频 |
| M4 | M3 世界物体、M1 冻结背景、M2 参考图与 Mask | `outputs/m4/object_refined.ply`、`scene_final.ply`、checkpoint、训练日志、多视角渲染及视频 |

完成状态及文件 SHA256 在 `records/m*_status.json`；命令和异常在 `logs/m*.log`。目录存在不代表阶段完成。安装、单元测试及输入准备不能当作生成结果。

完成或下载结果后，可用标准 Python 执行 `python industrial_swap_v2/scripts/verify_outputs.py` 核对四阶段全部登记产物的大小和 SHA256；它同时支持服务器与本机目录。单阶段使用 `--stage m3` 等选项。

原输入视频长约 **174.897 秒**，已有重建包含 350 个注册相机。轨迹展示视频使用全部 350 帧，以 12 fps 播放约 29.17 秒，属于加速展示，不是截取原视频前 29 秒。环绕视频为 72 个新相机视角。`renders/cameras.json` 保存每一帧的完整相机参数。

M1 直接复用已有 LangSplat 结果，验证源 PLY SHA256、索引完整分区以及点数。原场景 1,828,777 点，旧目标 31,352 点，保留背景 1,797,425 点。

M2 默认复用 AVP。`reuse_anchor: false` 会在独立目录调用现有 AVP 算法，重定向旧实现中硬编码的记录路径。深度采用现有 GSLE 体渲染加权欧氏距离及反向归一化；编辑区域内部深度置零表示未知。Qwen 收到的深度是**视觉参考，不是 ControlNet 条件**。位置提示图、背景深度及原图一并送入官方多模态接口。SAM 使用图像自动提示和 OpenCLIP 排序，最终 Mask 由 SAM 产生，没有手工坐标或人工资产。

M3 固定官方仓库 commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`、`microsoft/TRELLIS.2-4B`，默认 `1024_cascade`、low_vram，保留官方采样器配置。模型及依赖 snapshot 的 revision 单独记录。官方原生 Mesh 为 Z-up，官方 GLB 导出已转换到 Y-up `(x,z,-y)`；高斯化从 GLB 读取，避免二次坐标翻转。按面面积采样 200,000 个点，插值 UV 基色，按面切线/法线建立各向异性高斯，输出 Graphdeco log-scale、wxyz 四元数、opacity logit、SH0。原始 GLB 保留 PBR；Gaussian 基色不包含完整的金属/粗糙度着色模型。姿态适配仅拟合统一尺度、绕竖直轴旋转和地面平移，使用真实 Anchor Mask 自动拟合投影。

共享 A6000 具有 48GB 物理显存。当前资源策略要求至少 18GiB 空闲，给 Torch 分配器设置 16GiB 上限，使用官方 low_vram，不改变分辨率/采样器。CUDA 扩展的工作区可能在 Torch 分配器之外；实际峰值与失败记录需一并检查。若不足则保留 checkpoint，不降低核心推理配置冒充完成。

M4 直接读取 M3 PLY，复用 `gsle/diffusion.py` 的九通道 Inpainting UNet、ControlNet Depth 和 **text_only** CFG；不实例化 Zero123。固定教师与背景，默认严格保持新物体位置、尺度、旋转，只优化 RGB/opacity。参考 RGB/Mask 损失使用白底前景，DI-SDS 使用联合场景 RGB、深度、区域 Mask 和视角文本。有效批量 4、microbatch 2、2,000 次更新；每 100 步保存参数、Adam 状态、随机状态和输入签名。M4 要求 24GiB 空闲并限制 Torch 分配器为 20GiB；实际 GPU 通过 SWAP_GPU 选择。每次合成校验背景逐字节不变，M4 导出额外校验几何字段不变。

## API 调用限制与断点恢复

本次用户限制为只生成一张图。`qwen_client.py` 强制 `qwen-image-2.1-pro`、`n=1`，无负提示参数、无 POST 重试。Key 优先来自 `DASHSCOPE_API_KEY`，否则读取配置明确指定的受保护服务器文件，**整个 token 原样读取，不能按字符子串提取**。Key 不写入日志、Git 或本机。

提交前独占创建 `records/qwen_generation_once.lock`。超时或失败也保留锁；只有收到新的明确用户授权后，才允许独立的 `scripts/authorize_one_retry.py` 归档旧记录。流水线永远不会调用该授权脚本。成功后复用响应和 `edited_scene.png`，下载中断只恢复 GET，不重复生成。

实际排查中前两次鉴权失败来自最初 Key 解析代码截断带句点 token，已修复并添加回归测试。两次都返回 401，未获得图片；第三次在用户再次明确授权后成功，返回 1 张 1024×1024 图。所有尝试和授权记录保留。不能把三个提交记录表述为只提交过一次。

TRELLIS 保存原生 Mesh/voxel checkpoint；导出中断可读取该文件继续，避免重新三维生成。M4 读取 checkpoint_latest.pth 恢复训练。更改输入或主要配置时必须核对签名，不可把旧结果作为新实验。

## 环境

```bash
bash industrial_swap_v2/scripts/setup.sh
source scripts/common.sh
envs/download/bin/python industrial_swap_v2/scripts/download_models.py
envs/download/bin/python industrial_swap_v2/scripts/download_dependencies.py
```

独立环境为 `industrial_swap_v2/envs/trellis2`，Python 3.10、Torch 2.6/CUDA 12.4、flash-attn 2.7.3；模型、pip 缓存、源码、编译临时文件均留在新目录。使用本机代理通过 SSH 转发至服务器 17891。DINOv3 和 RMBG-2.0 的官方仓库可能要求 Hugging Face 访问授权；缺少授权时明确报错，不替换权重。旧 GSLE/SAM 环境只读复用。

实际接入中，用户指定可复用 `/mnt/sda/ubuntu/lkx/projects/TRELLIS.2`。已找到对应 `/mnt/sda/ubuntu/anaconda3/envs/trellis2` 和用户 HF 缓存。当前配置改用 `envs/trellis2_reused`，通过 `setup_reuse.sh` 创建独立 venv，只读继承上述已有环境的包；不向原环境安装依赖。官方源码仍用本项目固定 commit。现有主模型和 DINOv3 的 revision 与本次下载目标相同；RMBG 使用用户已运行过的官方 snapshot `5df4c9c76d8170882c34f6986e848ee07fd0ba43`。逐文件哈希与只读路径在 `records/reused_model_integrity.json`，实际包版本在环境记录中。所有新输出和新增包仍写入 v2。

若全新安装而不是复用，运行 `setup.sh` 后将 `project.trellis_python` 改为 `industrial_swap_v2/envs/trellis2/bin/python`。下载脚本固定到本次官方模型 snapshot。历史环境的 CUDA 扩展构建目录 `/tmp/extensions` 已被清理，无法追溯其中部分二进制的原始 Git commit；环境记录保留了实际包版本和源码/二进制文件 SHA256。当前服务器可按复用路径继续复现；全新编译环境需重新验证扩展兼容性。

## 来源及实现边界

- [Qwen 官方图像编辑 API](https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-image-editing)
- [TRELLIS.2 官方代码](https://github.com/microsoft/TRELLIS.2)
- [TRELLIS.2 官方模型](https://huggingface.co/microsoft/TRELLIS.2-4B)
- 原项目 GSLE 属于论文推导实现，公开论文未提供官方完整代码；本次复用其 DI-SDS，不将本项目声称为 GSLE 官方实现。
- Mesh→Gaussian、刚体尺度适配、冻结几何的外观优化是本次 v2 明确采用的接口适配。没有额外三维生成网络、消融实验或评价指标。

实际运行进度以服务器的 receipts 和日志为准。CPU 合成几何测试只验证坐标、材质转换和背景不变性，不作为机械臂生成结果。

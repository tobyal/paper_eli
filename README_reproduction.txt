工业场景 3DGS 编辑复现实验

工作目录：/mnt/sda/ubuntu/lkx/papers/paper_eli
SSH：ubuntuA6000，用户 kxliu
本地输入：E:\codex_projects\paper_eli\20261016-142321.mp4
输入 SHA256：061413776d7e163c53aa37cd556018bd65356df2128f4929a55ed90645c67116

统一编辑指令：将画面中央设备的外壳改成红色，保持其他物体不变。
执行英文：Change the housing of the industrial machine in the center to red, keeping all other objects unchanged.
具体目标：75秒参考视角中，黑色组件下方的白色设备底座/外壳。
目标身份、三维范围和人工检查证据保存在 records/target_*.json。

一、使用已经配置好的服务器重新运行

下载和安装时，保持本机代理127.0.0.1:17891运行，并保持配置RemoteForward的SSH连接开启。
已缓存模型的离线重跑无需联网。
额外SSH命令使用 -o ClearAllForwardings=yes，避免重复占用远程代理端口。
登录服务器后：

  cd /mnt/sda/ubuntu/lkx/papers/paper_eli
  source scripts/common.sh
  bash scripts/replay_baseline.sh repeat1

repeat1必须是尚不存在的输出标签。该脚本依次执行GaussianEditor、DGE、
DreamCatalyst、EditSplat，每个方法结束后检查PLY并按同一相机轨迹渲染。
新结果位于outputs/<method>_repeat1，缓存独立，不覆盖本次结果。
指标在records/replay_repeat1_metrics.json。完成运行后还应查看真实RGB，
程序退出码为0并不代表目标被正确改红。

单独运行和检查：
  envs/download/bin/python scripts/run_method.py dge --output-tag repeat2
  envs/download/bin/python scripts/finish_method.py dge
  envs/download/bin/python scripts/compact_progress.py

二、固定的数据和评价协议

视频174.89655秒，2560x1440，29fps。以2fps抽取350张1920x1080图像。
COLMAP先顺序匹配，再补充全量匹配，350/350视角成功注册。
使用原版2023 Graphdeco 3DGS训练30000步；按文件名排序，每8张取1张测试，
训练306张、测试44张。基础模型包含1828777个Gaussian。
基础30k模型在原版训练器报告的测试PSNR为28.454369dB。

编辑输入为经过内参同步调整的512x512中心裁剪；所有方法共用同一训练池。
各方法保留官方的视角采样策略：GE/DC最多96张，DGE每次20张，EditSplat使用
完整训练池。测试视角不参与编辑。训练步数和核心参数见run_method.py及每次运行
保存的argv/config；GE2500步、DGE1000步、DC1500步、EditSplat10个epoch。

目标候选选择：官方LangSAM产生候选mask，用已检查的三维目标投影选择同一台
物理设备。投影只选择候选，不生成替代mask。GE/DGE/DC使用官方Gaussian语义追踪，
阈值统一0.5。选择阈值的诊断、失败尝试和实际覆盖率均保留。

最终使用同一个原版3DGS渲染器渲染350个完整厂房视角，960x540。
视频12fps，共350帧，相对采集时间加速6倍。44个测试视角中24个目标可见，
目标红色比例、CLIP、背景PSNR/SSIM/LPIPS、深度重投影误差均有逐视角记录。
没有编辑后的真实GT；这些指标仅为代理指标，必须结合RGB结果判断。
本服务器为共享GPU，时间包含模型加载、分割、编辑、优化和方法自带测试。
每5秒记录GPU资源，单独记录进程显存峰值；不宣称为独占硬件速度排名。

三、源代码和环境重建顺序

此流程面向同一服务器，使用现有CUDA11.8、COLMAP3.12dev、ffmpeg、aria2c和conda。
scripts/common.sh中的ROOT及工具链路径应与当前服务器匹配。
官方仓库提交固定在scripts/00_bootstrap.sh；基础3DGS提交固定在03脚本。
每个环境的实际完整包列表在records/<method>.pip.freeze.txt。
模型快照和大文件SHA256在records/*model_downloads.json，下载器固定revision。

  source scripts/common.sh
  bash scripts/00_bootstrap.sh
  bash scripts/03_reconstruction_source.sh

创建下载环境（若缺失）：
  /mnt/sda/ubuntu/anaconda3/envs/lkx_spags/bin/python -m venv envs/download
  envs/download/bin/python -m pip install huggingface_hub==0.25.2

下载模型：
  envs/download/bin/python scripts/download_hf_parallel.py core
  envs/download/bin/python scripts/download_hf_parallel.py aux
  envs/download/bin/python scripts/download_hf_parallel.py editsplat_safety
  envs/download/bin/python scripts/download_hf_parallel.py imagereward
  envs/download/bin/python scripts/download_hf_parallel.py bert_safe
  envs/download/bin/python scripts/download_aux_models.py
  envs/download/bin/python scripts/04b_lpips.py
  envs/download/bin/python scripts/04c_clip.py
  envs/download/bin/python scripts/verify_offline_cache.py

run_method.py在重跑时启用HF/Transformers离线模式，使用已经下载并核验的固定缓存。

安装环境：
  bash scripts/02_environment_ge.sh
  bash scripts/05a_reconstruction_environment.sh
  bash scripts/06_environment_dge.sh
  bash scripts/06b_verify_dge.sh
  bash scripts/07_environment_dreamcatalyst.sh
  bash scripts/08_environment_editsplat.sh
  bash scripts/08c_check_editsplat_models.sh

应用已记录的兼容性/场景补丁（幂等脚本，不要重复叠加git diff文件）：
  envs/download/bin/python scripts/patch_local_model_paths.py
  envs/download/bin/python scripts/patch_instance_selection.py
  envs/download/bin/python scripts/patch_dge_angle_domain.py
  envs/download/bin/python scripts/patch_editsplat_logging.py

从视频重新预处理和重建（不要在现有结果目录内同时执行第二套重建）：
  bash scripts/01a_colmap_runtime.sh
  bash scripts/01_preprocess.sh
  bash scripts/01b_complete_matching.sh
  bash scripts/01c_finalize_sfm.sh
  bash scripts/05_reconstruction.sh

目标标注可用已保存的records/target_geometry.json，且需使用匹配的原始COLMAP模型。
若重新估计COLMAP坐标系，则重新运行05b/05c、derive_target_geometry.py并检查投影；
不能把旧坐标的标注直接用于新坐标系。之后运行05e_finalize_baseline.sh，检查评估mask。

四、已知适配与诊断

GE旧LangSAM的浮动依赖与当代依赖冲突：只固定官方依赖提交和打包引用。
DGE官方Config中default_factory误写为list：改为返回同一个list的lambda。
DGE相机角度浮点值略大于1产生NaN：仅补上arccos输入[-1,1]限制。
DC CUDA扩展setup.py的extracompileargs拼写错误：改为extra_compile_args。
ES本地路径适配只使用已核验的同一官方模型，不替换编辑模型。
ES额外保存二维编辑/MFG图、ImageReward排名和attention权重；只增加诊断输出。
ES按可执行源码默认filtering_ratio=0.85保留排名前15%的视角；README文字与参数默认
存在出入。AGT的k_percent=0.15在源码中除以100，实际首次裁剪比例为0.15%。
三维实例选择用于解决厂房中多台相似设备的身份歧义。
所有补丁、失败日志和中止原因保存在records和logs。

额外的提示词及目标中心视图实验只用于诊断，没有用于主对比；它们仍会出现
目标不改色或错改前景工作台的情况。主对比保留用户确认的同一指令和原始输入协议。
语义mask预览中的红色覆盖层是标注，不是编辑结果。

最终结果位置：outputs/<method>/artifact_manifest.json、common_render/multiview.mp4。
对比：outputs/comparison/，详细指标：records/common_metrics.json。
结果解释：REPORT.txt；机器可读汇总：records/final_comparison.json。

本地交付包reproduction_results.tar.gz包含复现脚本、实际代码补丁、环境清单、
运行日志、逐视角指标、四方法视频、对比图和44个测试视角的真实RGB渲染。
同时包含源模型的对应渲染、深度/透明度和目标mask，便于核查评价。
完整350视角渲染、编辑后的PLY、训练checkpoint和预训练权重保留在服务器；
路径及模型SHA256记录在各方法artifact_manifest.json与model_check.json中。

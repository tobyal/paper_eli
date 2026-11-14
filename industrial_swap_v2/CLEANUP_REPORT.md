# 四模块清理报告（2026-11-06）

已同步本机与 ubuntuA6000 的 `/mnt/sda/ubuntu/lkx/papers/paper_eli/industrial_swap_v2`，Web 已重启并完成只读接口检查。本次未提交或推送 Git；源码改动保留在工作区。原 `gsle/` 未修改。

## 正式结构

- M1：`modules/m1_localization.py`，复用 LangSplat 定位和高斯分区。
- M2：`modules/m2_image_generation.py`，复用 AVP、深度参考、Qwen 和 SAM；Web 图片上传/选择/缓存保持。
- M3：`modules/m3_trellis_generation.py` → TRELLIS、Mesh-to-Gaussian、原空间对齐 → `core/m3_composition.py` 原背景合成。
- M4：`modules/m4_refinement.py` → `core/multiview_masks.py` 双 Mask/遮挡候选准备 → `core/local_background_optimizer.py` → `core/m4_outputs.py`。设备全参数固定，仅优化允许的局部背景 SH0/透明度，250 步默认，200–300 步范围。
- `core/workflow.py` 统一新实验配置迁移/执行校验；`core/preview_panels.py` 仅承担已有图像排版。CLI 与 Web 使用同一 M4 入口。

## 清理范围

正式 M4 删除 DI-SDS、扩散教师初始化、设备外观训练与后续阶段分派。默认配置删除扩散采样/损失、2000 步、batch/microbatch、设备学习率、旧背景删除选项及无效参数。旧 bbox RGB 监督、耦合背景优化类和优化器分支已移出正式代码，只留下 union 实现。

`old_projection` 等少量缓存字段别名和 `scene_visibility_optimized.ply` 继续保留以读取已完成 union 结果；它们不代表旧算法或额外训练阶段。`di_sds_executed: false` 是保留的审计字段。

以下历史文件已归档（README 在正式位置重写，其余 30 个路径移出正式目录）：

- `repair_background.py`
- `inspect_background_repair.py`
- `export_background_repair.py`
- `show_background_repair.py`
- `run_background_repair.sh`
- `diagnose_placement.py`
- `show_placement_diagnosis.py`
- `core/object_occlusion_filter.py`
- `modules/m3_occlusion_filter.py`
- `scripts/run_multiview_refinement.py`
- `scripts/check_refinement_checkpoint.py`
- `scripts/inspect_background_support.py`
- `scripts/plot_refinement_metrics.py`
- `scripts/validate_multiview_refinement.py`
- `scripts/run_oagof.sh`
- `scripts/show_oagof.py`
- `scripts/validate_oagof.py`
- `scripts/export_supervision_audit.py`
- `scripts/compare_stages.py`
- `tests/test_object_occlusion_filter.py`
- `tests/test_multiview_refinement.py`
- `webapp/audit_oagof.py`
- `webapp/validate_oagof_reuse.py`
- `webapp/register_multiview_result.py`
- `BACKGROUND_REPAIR.txt`
- `OAGOF.txt`
- `MULTIVIEW_REFINEMENT.txt`
- `BACKGROUND_ONLY.txt`
- `README.md`
- `webapp/validate_live.py`
- `webapp/record_validation.py`

另外保存清理前混合实现/配置/测试的源码快照：

- `core/multiview_masks.py`
- `core/local_background_optimizer.py`
- `core/visibility_loss.py`
- `core/m3_composition.py`
- `modules/m4_refinement.py`
- `configs/default.yaml`
- `scripts/run_background_only.py`
- `tests/test_background_only.py`

以上归档位于 `legacy/pre_four_module_cleanup/`，`provenance.json` 提供逐文件 SHA256；`.gitattributes` 禁止对这些快照做换行转换。归档保留原导入和路径，不作为现工程可直接运行的入口。需要执行旧方法时，用完整历史提交 `071e179d1d63fed4605d1108f17cadfe293b5647` 建立独立 checkout 与输出目录。

主要接口修改：`core/context.py`、`core/workflow.py`、`core/m3_composition.py`、`core/multiview_masks.py`、`core/local_background_optimizer.py`、`core/visibility_loss.py`、`core/m4_outputs.py`、`modules/m3_trellis_generation.py`、`modules/m4_refinement.py`、`scripts/run_background_only.py`、默认配置、Web service/adapter/worker/app/viewer 与文档/测试。

## 数值行为与兼容性

旧/新 Mask 阈值、并集补集定义、小幅边界保护、三维支持、视角投票、梯度权限、损失、优化循环和检查点选择保持已有 union 数值行为。AST 回归检查关键函数和整个训练循环与归档一致，有效优化默认值逐项一致。

新 M4 的导出不再依赖专用脚本必须预先生成的 `source_preservation.json`，使用多视角清单中已保存的源文件哈希，旧专用清单存在时仍额外校验。最终场景显式读取经 SHA256 校验的优化背景，并原样复制设备文件；最后一步候选与通过约束的最终场景分别标注。

Web 新建任务分别采用 `web-v6-m3-composition` 与 `web-v6-m4-dual-mask` 缓存版本；M4 的步数实际写入 `local_background.iterations`，Mask 配置进入 M4 缓存签名。旧图像/模型素材和 TRELLIS 生成缓存仍可复用。复制历史实验时保留素材及 M1/M2，清除旧 M3/M4 阶段头；已有实验/记录不重写。旧 bbox Mask 不能作为新监督，复用旧设备时的新 Mask 写入当前 M4 目录。

历史 CLI 配置仍可读取，但不允许直接执行旧 DI-SDS/删除流程；应使用新配置或 `run_background_only --source-config ... --destination NEW_DIR`。旧长程优化无法以新 M4 恢复；当前短程中断保留现场，需新建运行。已完成优化但导出中断可继续导出。

## 仍需保留的依赖

相机、高斯数据、SH 和 raster 工具，以及既有 AVP/SAM 适配仍只读复用 `gsle/`/原环境。M1、M2、M3 的模型和环境依赖不变。现有环境中扩散包和权重未删除，以保护历史实验；正式 M4 不导入或初始化它们。Web 为历史结果保留旧设备文件名/进度记录读取兼容，均不提供旧算法执行入口。

## 验证结果

- 23 项核心 CPU 测试通过：双 Mask、权限/梯度、固定设备、损失/训练循环等价、配置迁移、缓存兼容、最终背景合成、Key/Qwen 一次性策略（模拟请求）。
- 21 项 Web CPU 测试通过：隔离数据库、图像上传/复用、不调用 Qwen 的 RGBA 导入、素材去重、缓存、M4 步数传递、旧过滤参数拒绝及历史实验复制。
- 2 项几何/素材 CPU 测试通过：Mesh 转换与无损合成；现有世界高斯复用不启动模型进程、不改变变换。
- 54 个正式 Python 文件语法检查通过；前端 app.js/viewer.js Node 语法检查通过；服务器 `git diff --check` 通过；两个 CLI `--help` 正常。
- 只读检查既有 250 步实验：41 个相机缓存满足准确的旧/新 Mask 与环境补集；20 万设备高斯文件哈希一致，最终场景的背景和设备行与源数据一致；789 个实验文件大小/修改时间不变。清理后的优化入口以缓存命中返回，renderer/Mask 重建入口被 mock 禁用，没有执行优化。
- 部署前后核对 3,450 个历史输出、记录与 Web 实验文件，大小/修改时间均不变。
- 重启后 6 个 GET 接口检查返回 200：health、实验列表、素材库、viewer catalog、app.js、viewer 页面；现有 12 个实验可读取，新的 M4 操作默认 250 步。

本次 Qwen 请求 0 次，TRELLIS 推理 0 次，训练/优化步数 0，GPU 渲染 0。未发现新的接口阻断。没有进行新的视觉质量实验。

既有 250 步结果的环境保持检查失败、最终选中 step 0 的状态保持原样。第 250 步真实候选仍单独保存；本次清理不将其标为通过检查的最终修复结果。

已有结果的机器可读检查：`records/maintenance/cleanup_20261106_existing_result.json`。

## M4 职责归属修正

双 Mask 及遮挡候选属于 M4，配置移至 `m4.multiview_supervision`，新数据写入 `m4/multiview_supervision/`。M3 仅生成/复用三维资产、空间对齐和场景合成，不再构建监督，也不依赖 M4 的 Mask 或优化参数。旧 M3 缓存校验后可只读复用，不搬移或重写历史实验。新增回归测试覆盖 M3 无监督构建、M4 输出位置、旧缓存兼容，以及 Mask 参数仅影响 M4 缓存。算法数值与训练循环未改。

职责归属修正后，25 项核心、23 项 Web、2 项几何测试（共 50 项）通过；41 相机既有缓存只读审计通过；服务器部署前后 3,451 个历史文件保持不变，Web 健康/实验/素材/查看器/前端接口正常。仍未执行模型生成、渲染或任何优化。新增只读审计记录为 `records/maintenance/m4_supervision_ownership_20261106.json`。

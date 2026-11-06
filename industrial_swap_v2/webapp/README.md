# Industrial Swap Web 实验工作台

在现有 M1–M4 算法之外增加实验版本、素材库、材料导出和持久任务调度。FastAPI + SQLite/SQLAlchemy 后端；前端使用原生 JavaScript/CSS，无 Node 构建依赖。GPU 任务只在 `ubuntuA6000` 上运行。原有 `industrial_swap_v2/run.py` 保持可用。

## 启动与访问

服务器首次安装：

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
source scripts/common.sh
envs/download/bin/python -m venv industrial_swap_v2/envs/webapp
industrial_swap_v2/envs/webapp/bin/pip install -r industrial_swap_v2/webapp/requirements.txt
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.manage init
bash industrial_swap_v2/webapp/start.sh
```

`init` 创建数据库，并校验、登记当前已完成的历史 M1–M4 产物；不会重新生成，也不会写入旧输出目录。重复执行不会重复导入。`start` 启动独立 HTTP 服务和持久 worker。服务默认只监听服务器 `127.0.0.1:18086`，通过 SSH 访问，不公开到公网。

Windows 本机连接：

```powershell
cd E:\codex_projects\paper_eli
powershell -ExecutionPolicy Bypass -File industrial_swap_v2/webapp/connect.ps1
```

保持该窗口运行，在浏览器打开 **http://127.0.0.1:18086**。当前会话已建立该转发时，无需重复启动。不能同时使用 `ClearAllForwardings=yes` 和 `-L`，否则本地 Web 转发也会被清除。

查看服务或更新代码后重启：

```bash
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.manage status
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.manage restart
```

重启只操作本程序 PID 文件且命令行匹配的服务进程。正在执行的 GPU 子进程独立运行，新 worker 检查 PID 和完成标记，不会重复提交同一任务。GPU 子进程本身失败后，页面提供“恢复此运行”，沿用原目录和检查点。不要直接修改正在运行的输入配置。

## 操作步骤

1. **创建或复制实验。** 当前注册的是已有工业场景、黑色组件及白色底座定位。新设备描述可修改；复制实验引用已有的不可变结果。更改描述会使 M2 及后续结果失效。
2. **M1。** 复用已有 LangSplat 结果，或调用原有 M1 代码重新整理目标/背景分区。页面显示点数、Bounding Box、生成区域与参考相机。这里没有新增 LangSplat 训练算法；更换待移除对象前，需按原项目流程提供对应定位。
3. **M2 准备材料。** 复用原 `prepare-only` 与同一 `build_prompt()`，下载 Anchor、位置提示、Mask、完整/背景深度、Prompt、相机和条件 JSON，支持单文件与 ZIP。
4. **M2 选图。** 上传外部 `edited_scene` 会运行现有自动 SAM；上传 `object_rgba` 则保留 Alpha 并生成 Mask/白底图，跳过 SAM。也可直接选择图库中已有图像。只有明确点击并确认“调用 Qwen API”才提交一次请求；导航、上传、材料准备及后续阶段均禁止调用 Qwen。
5. **M3。** 可运行官方 TRELLIS.2，或选择/上传已有三维资产。GLB/Mesh 只做高斯化、对齐和合成；局部 PLY 跳过高斯化；世界 PLY 校验场景与变换后合成，也可先反变换到局部空间重新对齐。默认重新对齐。
6. **M4。** 配套 M1 背景、M2 前景/Mask/相机和 M3 设备齐全后，启动原有 DI-SDS。默认 2,000 步，背景和几何冻结，可查看日志、恢复失败运行、下载最终 PLY 和视频。

图片必须保留对应 Anchor 的完整画布、视角与设备位置；不同分辨率允许等比例变化。已居中裁切的产品图片不能直接冒充场景相机参考。界面要求用户确认画布，并验证比例、场景与相机元数据。这里只能验证声明与元数据，不能自动证明任意外部生成图的透视正确。

现有 SAM 自动提示针对**黄色设备**。其他颜色或类别可导入已分割 RGBA；本次没有修改分割核心算法。深度仍是 Qwen 的额外视觉参考，不是原生 ControlNet 条件。

## 三维素材约定

| 类型 | 必要信息 | 实际执行 |
|---|---|---|
| Mesh | 自包含 GLB；独立 OBJ、STL、带面的 PLY；局部 Y-up | 原 `convert()` → `fit()` → `compose()` |
| 局部 Gaussian | Graphdeco SH3 PLY，局部 Y-up | 原 `fit()` → `compose()` |
| 世界 Gaussian | 同格式 PLY，来源场景，原 `scale / rotation / translation` | 同场景直接合成，或逆变换后重新对齐 |

上传不接受带外部文件/网址引用的 GLB 或带外部 MTL 的 OBJ；这类模型请先导出为自包含 GLB。Gaussian PLY 会检查字段、有限值、旋转和尺度，并统一字段顺序。普通点云不是 Gaussian PLY。Checkpoint 仅登记本程序生成的可信文件，不执行用户上传的 Python/Pickle。

三维素材记录输入图像哈希/素材 ID、模型、变换、预览和使用实验。原生 TRELLIS checkpoint 保留在对应运行目录及素材登记中。当前版本提供静态预览和下载；没有增加浏览器 GLB 交互查看器。

## 数据和缓存

```text
webapp/data/
  database/app.db               # SQLite WAL；metadata only
  assets/<sha-prefix>/<sha.ext> # 内容寻址文件；相同 SHA256 去重
  experiments/<id>/runs/<id>/
    config.yaml                # 不向前端暴露的执行配置
    job.json
    authorization.json         # 仅显式 Qwen 操作创建
    execution.log
    outputs/m1 ... m4          # 当前阶段新结果；依赖阶段为只读引用
    records/                   # 原阶段 receipt、API 锁、M4 进度
    logs/
  service/                     # Web/worker PID、日志和单 worker 锁
  validation/                  # 真实端到端验证记录
```

数据库表：`experiments`、`assets`、`experiment_assets`、`stage_runs`、`generation_records`，以及记录去重上传来源的 `asset_events`。大文件不进入数据库。`data/`、模型、密钥和产物不进入 Git。

每次运行固化输入运行 ID、素材版本、参数与配置。切换 M2 素材只清空当前 M2/M3/M4 指针；切换 M3 素材只清空 M3/M4 指针。历史运行和文件保留。复制实验共享素材与完成结果，后续运行仍有独立目录。

阶段缓存同时使用参数签名和上游实际产物的 SHA256，并在命中时复核登记文件的大小与哈希。即使生成参数相同，生成图片不同也不会误用旧阶段结果。恢复部分完成的运行时继续原适配器，不会把残留文件标记成缓存命中。TRELLIS 另按前景图像哈希、模型/代码配置及 seed 复用已有 Mesh，再针对当前场景对齐。不会把安装、排队或失败标记为完成。

Qwen 每次明确操作具有幂等请求编号、单独授权记录和独占磁盘锁。超时、401 或失败也保留锁；恢复只可利用已有响应继续下载/分割，不能自动重发。测试与素材复用模式设置 `SWAP_DISABLE_QWEN=1`，直接在原客户端入口拒绝提交。Web 前端不读取或持有 API Key。

Worker 串行调度本应用任务，按空闲显存选择 GPU，保持原训练的显存预算。TRELLIS 推理要求 18GiB 空闲、M4 要求 24GiB；仅复用三维资产时只为渲染预留 4GiB，M1 与 RGBA 导入不等待训练级显存。资源不足保持排队；不会停止其他用户进程。日志轮询与文件下载独立于 GPU 子进程。

## 验证命令

```bash
# 隔离测试数据库，CPU only；不启动模型或真实 API
industrial_swap_v2/envs/webapp/bin/python -m unittest discover -s industrial_swap_v2/webapp/tests -v
industrial_swap_v2/envs/trellis2_reused/bin/python -m unittest discover -s industrial_swap_v2/tests -v

# 真实验证：会实际运行一次 TRELLIS.2 和 2,000 步 M4，不调用 Qwen
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.validate_live
# 排错后恢复原任务：追加 --resume-failed

# 真实 GLB 上传 / 局部 PLY / 世界 PLY 路径验证
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.validate_asset_routes

# 汇总实际运行、文件哈希、耗时、测试结果与代码版本
industrial_swap_v2/envs/webapp/bin/python -m industrial_swap_v2.webapp.record_validation
```

真实验证使用已有的 Qwen 场景图作为上传测试文件，不宣称重新从其他平台生成了图像。所有新结果留在 Web 数据目录；验证脚本在结束时重新计算历史产物 SHA256，确认原实验未被覆盖。

`webapp/records/validation.json` 汇总实际任务状态与产物校验，包含阶段缓存、TRELLIS 自动模型缓存和两种 PLY 复用路径。`tests_web.log` / `tests_cli.log` 为测试输出，`webapp.pip.freeze.txt` 固定 Web 环境依赖。网页上传 RGBA、复制实验、日志及结果预览另经过真实浏览器操作检查。验证记录只有在全部实际运行和哈希检查通过后才标为 `complete`。

API 交互文档：`/api/docs`。主要接口为 `/api/experiments`、`/api/assets/upload`、`/api/experiments/{id}/selection`、`/api/experiments/{id}/runs`、`/api/runs/{id}/resume`、`/api/runs/{id}/bundle` 与日志/产物下载接口。写操作需要同源 `X-Swap-Action: 1` 请求头。本版本面向经 SSH 使用的单用户科研工作台，不提供公网多用户账户系统。

# M1 人工辅助案例准备

打开 `http://127.0.0.1:18086/m1-cases`，或从实验工作台侧栏进入 **M1 人工案例准备**。
本页只执行 M1。不会提交 Qwen、启动 TRELLIS.2、M2/M3/M4 或重新训练 LangSplat。

## 使用

1. 浏览或搜索 COLMAP 原始照片；没有 SAM/CLIP 缓存的照片标为不可选。
2. 点击设备，或拖框后勾选缓存 SAM 区域。可以合并同一尺度的区域；切换尺度会清空选择。
3. 查看自动推荐的五个英文描述及 CLIP cosine 分数，默认选第一条；也可切换或手动修改英文。
4. 点击运行 M1。三维查询在共享 GPU 上排队执行，HTTP 页面不阻塞。
5. 查看原图/参考 Mask、目标高斯高亮、独立目标及删除后的背景，并检查另外两个注册视角。
6. 满意后明确点击确认保存。定位记录与已确认案例分开，刷新后可从记录中恢复。不会自动凑满十例。

分数不是置信概率。候选词库以工业设备类别、颜色、结构描述覆盖场景；CLIP 不保证正确识别未覆盖类别。
SAM 区域来自已有自动分割缓存，框选并非重新运行 SAM，也不直接定义三维删除范围。
本次没有改变原 M1 包围盒删除逻辑，因此删除后的残留/误选仍应通过三视角人工检查。

## 实现与方法边界

- `core/m1_reference.py`：COLMAP 原始相机、SAM/CLIP 缓存、同尺度区域合并、真实余弦排序。
- `configs/m1_semantic_library.json`：通用工业设备词库；实际排名由选中区域的特征决定。
- `scripts/m1_language.py`：用原 LangSplat OpenCLIP ViT-B-16 / laion2b_s34b_b88k 编码文本；按选择重新查询已训练三尺度语言场。冻结模型，仅推理。
- `scripts/m1_assisted_localize.py`：沿用原语义阈值 0.5、透明度阈值 0.05、投影深度约束、可见种子分位数及 5% 边界、地面保护规则；调用现有 M1 无损高斯划分。
- `webapp/m1_cases.py`：输入快照、推荐、队列、查看及显式确认 API。
- `webapp/m1_job.py` / `m1_worker.py`：隔离解释器、共享原单 worker 队列、重启恢复；不导入后续生成模块。
- `webapp/static/m1_cases.*`：原生 Web 界面；现有素材库及实验流程保留。

合并区域使用各区域归一化 CLIP 特征的面积加权均值，之后 L2 归一化。只允许同尺度、不重叠的 SAM 标签合并，仍查询对应尺度的三维语言场。
三维 CLIP 原型相似度作为记录保留，不添加新的种子阈值。定位范围仍由原有文本/深度种子算法产生。
辅助视角从原始配准相机中选取，目标在视野内，且与参考及彼此有方向差异。

## 目录与后续复用

均位于服务器项目内 `industrial_swap_v2/webapp/data/m1_assist/`：

```text
vocabulary/              # 官方 CLIP 文本特征、词库及权重 SHA256
selections/<hash>/       # 原始照片、相机、同尺度 Mask、区域/合并 CLIP 特征、全部排序
jobs/<uuid>/             # 每次独立定位，不覆盖已有记录
cases/<uuid>/            # 明确确认后复制的独立 M1 案例
  reference/             # 照片、相机、Mask、CLIP、语义推荐排名及来源哈希
  semantic/              # 三尺度本次查询、权重来源、原场景顺序核验
  localization/          # target/edit_region、种子/删除/保留索引
  outputs/m1/            # target_gaussians.ply、background_gaussians.ply、manifest
  inspection/            # 同相机高亮/独立目标/背景及另外两个视角
  config.yaml            # 独立案例路径，最终选择的英文语义
  result.json            # 数量、诊断指标、三个相机
  integrity.json         # 全文件 SHA256
  records/               # M1 完成记录
  run.log                # 实际执行日志
validation_jobs/         # 操作测试专用，不计入正式案例
validation_cases/
```

案例配置可供后续现有 CLI 读取；M1 定位目录和输出目录已独立保存。`reference_camera.json` 只用于 M1 坐标系，不冒充 AVP：配置将未来 M2 的 `reuse_anchor` 设为 false。
确认保存会复制所需数据并改写内部路径，不依赖临时定位记录；保存前后校验哈希。主场景及预训练模型仍只读复用项目路径。

## 文本词库准备与测试

词库首次编码使用现有 LangSplat 环境和本地权重（无下载/训练）：

```bash
cd /mnt/sda/ubuntu/lkx/papers/paper_eli
# 先检查共享 GPU 空闲情况，再指定 CUDA_VISIBLE_DEVICES。
CUDA_VISIBLE_DEVICES=2 envs/gsle_langsplat/bin/python -m industrial_swap_v2.scripts.m1_language \
  --root "$PWD" --output "$PWD/industrial_swap_v2/webapp/data/m1_assist/vocabulary"
industrial_swap_v2/envs/webapp/bin/python -m unittest \
  industrial_swap_v2.tests.test_m1_reference \
  industrial_swap_v2.webapp.tests.test_m1_assist \
  industrial_swap_v2.webapp.tests.test_webapp
```

词库变更后需重新编码；服务器会拒绝使用与当前词库哈希不符的文本缓存。
UI 验证使用 `/m1-cases?validation=1`，测试数据明确隔离，不代表用户已选定论文案例。

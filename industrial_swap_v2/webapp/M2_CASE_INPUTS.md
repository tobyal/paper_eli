# 已确认 M1 案例的 M2 输入准备

入口：`http://127.0.0.1:18086/m2-cases`，或 M1 案例列表里的“准备 M2 输入”。

1. 选择一个已确认的 M1 案例，填写新设备英文描述。
   当前 10 条正式案例已各配 3 个新设备方案；原第 11 条重复定位已归档。点击“使用方案 A/B/C”会填写英文描述和默认 Prompt，可继续编辑。该操作只处理文本，不提交输入准备任务。
2. 点击“根据描述更新默认 Prompt”，检查并按需修改全文；也可直接点击准备自动使用默认文本。
3. 点击“准备 M2 输入”。系统校验真实 M1 文件，复制独立输入快照，运行已有 AVP 和 `core/depth.py`。
4. 查看 Anchor RGB、位置提示、原始深度、屏蔽编辑区的背景深度，以及保存的 Prompt。
5. 复制 Prompt 或下载完整 ZIP。修改描述/Prompt 后再次准备会生成新版本，不覆盖已有输入。

页面仅准备输入。它不重新执行 M1、不训练 LangSplat、不调用 Qwen、不运行 SAM/TRELLIS.2/M4。
正式案例与 `?validation=1` 测试案例分开显示。描述可由用户填写，也可选用按各案例照片和编辑空间编写的方案，不将历史黄色机械臂描述自动套到每个案例。
方案保存在 `configs/m2_case_description_options.json`，不写入或改变 M1 的不可变产物。原案例 02 与 11 的目标高斯索引相同；11 已从当前列表和批量任务排除，历史数据仍保留。

## AVP 与空间约束

默认复用原 100 轨道视角、8 次图像旋转的亮度分析及原可见度/覆盖率门槛。若轨道中没有合格视角，复用项目已有 `core/observed_anchor.py`，在已配准观察相机中使用相同评分及门槛自动选择，完整记录候选池来源；仍无合格视角时明确失败，不强行套用 M1 相机。

M1 参考照片只作为原定位信息保存。M2 Anchor 独立选择；RGB、Depth、位置提示及 Mask 均使用同一 Anchor 内外参和 1024×1024 画布。
位置提示为该案例三维编辑空间的投影；背景深度使用原始深度，并在这一投影内置零。零表示未知区域。
此任务不做地面补全或背景生成。

`conditions.json` 保存 M1 相机、M2 相机、三维编辑区域、AVP 选择、案例编号及输入文件 SHA256。
`generation_inputs.json` 明确 Qwen 三张输入图的顺序。

同一 M1 的不同描述可复用已完成且逐文件哈希通过的 AVP/RGB/Depth 条件。缓存必须匹配场景、随机种子、M2 条件配置及 M1 几何文件；新 Prompt 单独构建，条件来源写入 `conditions.json/condition_cache`。

批量准备入口为 `scripts/prepare_m2_description_batch.py submit`，进度用 `status` 查看，全部准备后用 `export` 校验并导出。对于实际因 AVP 无合格视角而失败的目标，显式执行 `review --case-number N` 会保留历史运行，在新版本使用已配准拍摄位姿的扩展画幅：保持 R/t，以目标投影为中心，按完整编辑空间及 35% 余量设置方形内参，重新渲染高斯，允许视线超出原照片边缘，不做二维补边。亮度评分和原 90% 可见度、75% 场景覆盖门槛保持不变；若仍无合格候选，只输出覆盖率合格且可见贡献最高的人工检查版本，记录 `quality_status=manual_review_required`，不能视作 AVP 质量通过。

高俯角候选曾用于诊断，但在 RGB 检查中出现明显拖影，没有纳入最终 30 组输入；诊断实现保存在 `legacy/diagnostics/m2_elevated_anchor_search.py`。

## 保存与后续导入

M1 原始案例保持只读。每个准备版本对应一个可独立继续的 Web 实验：

```text
industrial_swap_v2/webapp/data/experiments/<experiment_id>/runs/<run_id>/outputs/m2/
  anchor.png                 # Qwen 图 1：场景 RGB
  position_hint.png          # Qwen 图 2：三维编辑位置
  depth_background.png       # Qwen 图 3：屏蔽编辑区的背景深度
  depth.png                  # 检查用原始深度
  depth.npz                  # 原始数值 depth / alpha / 编辑 Mask
  edit_mask.png
  anchor_camera.json
  conditions.json
  generation_prompt.txt
  generation_plan.json
  generation_inputs.json
  avp_work/avp/              # 全部候选、评分和选中视角
```

ZIP 包含全部 Qwen 条件图、Prompt、相机、原始数值深度和输入说明。AVP 全量候选保存在服务器版本目录。

生成图片后点击“后续上传生成图 → 对应实验”，使用已有外部图片上传与素材库功能。图像需保留本版本 Anchor 的完整画布及相机视角；上传后仍需用户另行点击完成 M2 才运行前景提取，M3/M4 也不会自动启动。每个版本单独绑定相机，避免误用其他版本的素材。
这些案例实验的 Qwen API 提交被后端禁止，原有历史实验不受影响。

## 代码入口

- `webapp/m2_cases.py`：确认案例校验、只读 M1 快照、现有 `stage_runs` 队列桥接、独立版本查询。
- `core/m2_prompt.py`：Web 与 M2 共享的无 GPU Prompt 构建；手改 Prompt 原样保存。
- `core/depth.py`：复用 AVP、RGB/深度渲染与三维编辑 Mask；保留完整相机及案例来源。
- `modules/m2_image_generation.py`：仍使用 `prepare_only=True`，只写输入计划。
- `scripts/audit_m2_case_inputs.py`：逐像素检查三维区域投影、深度屏蔽、相机对齐、M1 完整性与未生成状态。

测试不提交 API。Web CPU 回归测试与 `tests/test_case_preparation.py` 使用隔离数据；实际渲染验证仅使用明确标记的验证案例。

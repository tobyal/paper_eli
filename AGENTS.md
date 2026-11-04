# Industrial 3DGS editing reproduction

- Execute experiments on SSH host `ubuntuA6000` as `kxliu`, under `/mnt/sda/ubuntu/lkx/papers/paper_eli`.
- Local workspace stores input video, reproducible scripts, logs, and downloaded previews. Do not run GPU training locally.
- Input: `20261016-142321.mp4`, SHA256 `061413776d7e163c53aa37cd556018bd65356df2128f4929a55ed90645c67116`.
- User-approved instruction: 将画面中央设备的外壳改成红色，保持其他物体不变。
- Run official methods in order: GaussianEditor, DGE, DreamCatalyst, EditSplat. Preserve their core algorithms and record every compatibility patch.
- Use the local HTTP proxy at `127.0.0.1:17891`, forwarded by the SSH host configuration to remote loopback port 17891. `scripts/remote.py` uses the existing tunnel, without claiming the port again.
- GPU resources are shared. Inspect availability; never stop other users' processes or modify their environments.
- Use isolated project environments and model caches. Save exact upstream commits, effective configs, commands, logs, timing, checkpoints, renders, and videos.
- Never label setup, queued jobs, or unsuccessful runs as completed experiments. Compare only actual outputs on common cameras and masks.

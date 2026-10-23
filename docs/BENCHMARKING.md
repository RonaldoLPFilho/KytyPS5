# 性能复测

正式入口见 [当前版本](CURRENT-PLAYABLE.md)，结论只记入 [实验简记](EXPERIMENTS.md)。

## 固定位置并保留进程

确认显示器活动且为 60 Hz、没有另一个模拟器进程。使用一个尚不存在的输出目录：

```bash
python3 tools/local/benchmark-entry.py \
  --config _Build/agent30fps-20260917/launch-wins7.json \
  --save-baseline _Build/walk-fps-20260914/save-baseline \
  --out _Build/experiments/new-benchmark \
  --walk-forward-seconds 10 --seconds 20 --keep-seconds 1800 --cpu-times
```

脚本备份用户存档、载入固定存档、正常 Continue，确认 HUD 后前进并稳定。等 `result.json` 为 `measured` 且 `retained=true`，检查截图确实是目标场景。加载、菜单、额外镜头输入不能算入静止 FPS。

## 同一进程 A/B

候选必须有可安全切换的运行开关，使用独立二进制和配置；不覆盖正式版。用相应构建的有效符号替换下列 `SYMBOL`：

```bash
python3 tools/local/abba-switch.py _Build/experiments/new-benchmark/result.json \
  --symbol SYMBOL --sequence 0,1,1,0,1,0,0,1 \
  --seconds 20 --settle 2 --out _Build/experiments/new-ab
```

工具核验 PID、启动时间、二进制 SHA 和符号大小。合并取总帧数/总时间，不平均窗口 FPS；保留全部窗口与范围，出现漂移要复测。首次验证结果正确，再关闭重型计数测性能。

- `abba-cpu.py` 补充渲染线程 CPU/帧，检查 FPS 量化掩盖的小变化。
- `quick-benchmark-game.py --pid PID --seconds 10 --camera-frames 0 --screenshots` 快速观察，截图在窗口外。
- `profile-game.py --pid PID --renderer --seconds 10 --label diagnostic` 仅作独立诊断。采样/捕获/输出校验的 FPS 不当作净收益。
- 实际 present 使用 `kyty_local_present_stats[0]`（先开启 present timing）；FIFO 下 `[7]` 的 free-run flip 数不能替代实际帧数。
- 比较开关还需考虑缓存/页保护等残留状态，不能假定关开关立即撤销所有副作用。离线预先给出未来输入的倍数不是游戏 FPS。

## 结束和清理

```bash
touch _Build/experiments/new-benchmark/stop
```

等脚本正常关闭游戏、`original_save_restored=true`，确认没有遗留进程。提取方案、A/B 数字、验证与结论追加到实验简记，再删旧截图、抓取数据、测试构建和日志。不要删除启动说明中的必要资产，也不要删除正在使用的运行目录/锁。

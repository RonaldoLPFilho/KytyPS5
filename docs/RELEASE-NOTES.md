## New in this build

- **Less memory, and it no longer creeps up the longer you play**:
  - After a save loads the emulator takes about 2 GB less RAM: the shader programs it keeps no longer have a second
    copy inside the graphics driver (on NVIDIA that copy alone was about 1.8 GB), and its shader preparation keeps
    more of its data compressed or on disk. On PCs with 16 GB of RAM it now needs about 14 GB after a save loads
    (it was about 20 GB).
  - Every trip to another world or area left about 25 MB behind for good, so long sessions kept growing. That is
    now released (tested with four rounds of travel through all five worlds and the Nexus: memory stays level after
    the first round, which still adds what the newly seen shaders need).
- **HDR output**: the launcher's "HDR" checkbox (or `run-windows.ps1 -Hdr` / `--hdr`) outputs HDR10 on an HDR display.
- **VSync option**: the launcher's "VSync" checkbox removes screen tearing (issue #12).
- **Fixes**:
  - Game version 1.05 crashed after about 10 minutes in Boletaria 1-1 (issue #10).
  - On GPUs with 8 GB of video memory, some very large textures could fail to load (issue #5).
  - A game folder with a very long path (for example deep inside OneDrive) could not be opened by the emulator or the
    shader precompile (issue #13).
  - A crash when travelling to the Tower of Latria by archstone.
- **Shader precompile**: the shader list now includes tours of every map, so more of the shaders a first visit needs
  are ready before you play.
- **Linux**: a Linux package is built with each release too (experimental, contributed in PR #3).

## 本版本的变化

- **内存占用更少，玩得越久也不再越涨越多**：
  - 读档后模拟器少占约 2 GB 内存：模拟器保存的着色器程序不再在显卡驱动里另存一份（NVIDIA 上光这一份就约 1.8 GB），着色器准备的数据也更多地压缩存放或放在磁盘上。16 GB 内存的电脑上，读档后现在约需 14 GB（之前约 20 GB）。
  - 之前每去一次别的世界或区域都会永久留下约 25 MB 的内存，所以玩得越久占用越大。现在这些都会被释放（测试：用石碑在五个世界和大厅之间来回走了四轮，第一轮之后内存不再增长；第一轮仍会增加新见到的着色器所需的内存）。
- **HDR 输出**：启动器里勾选"HDR"（或 `run-windows.ps1 -Hdr` / `--hdr`），在 HDR 显示器上输出 HDR10。
- **垂直同步选项**：启动器里勾选"VSync"可以消除画面撕裂（issue #12）。
- **修正**：
  - 游戏 1.05 版在波雷塔利亚 1-1 玩约 10 分钟后会崩溃（issue #10）。
  - 8 GB 显存的显卡上，部分特别大的纹理可能载入失败（issue #5）。
  - 游戏文件夹路径很长时（例如放在 OneDrive 很深的目录里），模拟器和着色器预编译打不开（issue #13）。
  - 用石碑前往拉特利亚之塔时崩溃。
- **着色器预编译**：着色器列表加入了所有地图的巡游记录，第一次进入各区域时需要的着色器更多地会提前准备好。
- **Linux**：每次发布也会附带 Linux 版本（实验性，来自 PR #3）。

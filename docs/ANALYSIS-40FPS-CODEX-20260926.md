# 40 FPS 优化路径分析

- **40 FPS 还需减少约 5.7 ms/帧；现有证据支持优先优化 GPU 与回读链，但不足以判定所有渲染 CPU 优化都已失效。**
- **第二个 Vulkan 队列不是已证实的突破口**：帧首剔除目前可争取的排队时间约 1.82 ms，而且客体标签的真实依赖尚未证明可绕开。
- 代码中的关键发现是：普通 EOP 标签通常在 CPU 解析时立即写入；异步回读入口也没有优先利用已有反馈快照。
- 推荐先做反馈路径与着色器的小规模验证，再做参数表批处理；多队列采用受限试点。**可以建立逼近 40 FPS 的路线，但目前不能承诺达标。**

已核对分支及提交为 `experiment/perf-40fps-20260926`、`ec59851`。本次仅阅读源码、已有文档及 Vulkan 规范，没有修改文件、编译或运行模拟器、游戏、测试。以下实验均为后续建议，未执行。

## 一、对关键路径判断的审核

### 1. 判断方向成立，但“渲染线程再快也没用”说得过强

你的实验很有价值：上传、BDA、回读槽三项合计减少约 0.8 ms 渲染 CPU，FPS 不变，说明**这些被削掉的工作没有推进当前帧的最终完成点**。

但它不能证明主帧前半段的 CPU 工作也没有价值：

- 主帧前半段 GPU 经常缺工作，存在 CPU 翻译限制。
- 后半段 GPU 已积压，此时减少 CPU 工作可能只增加提前量。
- 同样减少 1 ms，发生在这两个位置，帧时间收益可能完全不同。

还应修正“CPU 更早结束导致 GPU 尾巴更晚结束”的解释。源码中的 `LiveControl::Flip()` 在 CPU 解析路径执行，真正的 flip 完成通过 GPU tick 后的回调处理。CPU 提前到达测量边界，**即使 GPU 的绝对完成时间不变，相对 flip 的尾巴也会变长**。[CPU flip 位置](./src/graphics/guest_gpu/graphicsRun.cpp) 的具体位置为 [graphicsRun.cpp:1838](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/guest_gpu/graphicsRun.cpp:1838)。

因此，后续必须比较同一时钟下的：

> 重 GPU 批次首次提交 → GPU 开始/完成 → 回读可用 → 客体主提交 → 连续帧完成间隔。

不能只比较“相对 CPU flip 的尾巴长度”。

另外，`6.58 + 26 ≈ 32.6 ms`，并非 32.6 FPS 对应的 30.7 ms。这些数字可能来自不同采样或边界定义，不宜直接组成一个精确的加法模型。CPU 分类也存在包含关系，缓冲同步、BDA、memcpy 等不能再与 draw/dispatch 总时间全部相加。

### 2. 40 FPS 同时要求改善依赖延迟和 CPU 余量

| 项目 | 对目标的含义 |
|---|---|
| 当前约 30.7 ms | 需要减少约 **5.7 ms，18.6% 帧时间** |
| GPU 忙 17.75 ms | 达到 25 ms 时约占 71%；总 GPU 工作量本身没有排除 40 FPS |
| 渲染 CPU 23.9 ms | 距离 25 ms 只剩 1.1 ms；仍应争取减少 CPU 工作，为必要等待留空间 |
| 帧首剔除排队 1.82 ms | 即使完全消除，约为 **34.6 FPS**，单独不够 |
| 后处理链 2.3 ms | 即使提速一倍，也只省 **1.15 ms GPU 时间**，且未必全部转化为帧收益 |

所以合理路线是：**推进前半段 GPU 工作的供给，缩短后半段 GPU 工作，并减少反馈路径上的额外等待。**

## 二、剔除究竟在等什么

### 1. 源码确认了四种不同的顺序约束

| 约束 | 当前实现 | 对优化的影响 |
|---|---|---|
| 客体帧边界 | `CanProcessSubmission()` 禁止下一 epoch 越过当前边界；边界等待本 epoch 计算提交消费完 | 这是 CPU 命令消费约束，不等于 GPU 帧完成 |
| 客体 WAIT | `WaitRegMem()` 读取客体地址、按 func/ref/mask 比较，不满足就挂起 PM4 | 是否能提前执行，首先取决于这个等待的生产者 |
| 宿主执行依赖 | 各队列共用调度器，存在广泛的计算访问、写冒险和批次屏障 | 会把原本独立的工作串起来 |
| CPU 读结果 | 回读复制记录到当前命令缓冲，等待它的全局 tick，再发布客体内存 | 等到的可能是包含无关工作的提交前缀 |

相关实现见 [调度与 WAIT](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/guest_gpu/graphicsRun.cpp:367)、[帧边界限制](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/guest_gpu/graphicsRun.cpp:512)、[回读创建与完成](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/cache/bufferCache.cpp:107)。

### 2. 重要发现：标签值可见，不代表 GPU 已完成

`WriteAtEndOfPipe()` 的普通 32/64 位写入路径直接调用 `memcpy`。随后调用的 `Sync::WriteAtEndOfPipe*()` 主要记录调试信息，带中断的路径再安排延迟回调；并不是所有标签写入都延迟至 GPU 完成。GDS 读出等分支另有同步处理，不能混为一谈。[标签直接写入](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/guest_gpu/graphicsRun.cpp:1509)、[Sync 实现](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/sync.cpp)。

这意味着，对于普通可直接访问的标签页：

> WAIT 通过，只证明 CPU 看到了标签值；GPU 侧顺序仍可能由当前的提交顺序和屏障补足。

因此，不能把当前“WAIT 已通过”的计算命令直接送入第二队列。那会移除原先隐含承担正确性的宿主依赖。

### 3. 能确认的具体标签有限，不能据此认定依赖整个后处理尾巴

[RENDER-SPLIT-NOTES.md](/home/chen-xiao/git/KytyPS5-pr500/docs/RENDER-SPLIT-NOTES.md) 记录了：

- q9/q49 等下一帧计算等待地址 `0x24175da90`；
- 该值被识别为帧计数。

但现有文字记录没有给出该等待对应的完整 `ref/mask/func`、匹配的生产者包及其位置。通用 PM4 解析源码也不包含游戏运行时的这些值。

**目前不能确认：这个标签由哪个客体队列、哪种原始 EOP/RELEASE_MEM 写入，以及它的语义是否覆盖那 2.3 ms 后处理链。** 同样，没有证据能把剔除输入具体认定为上一帧深度、Hi-Z 或某个反馈缓冲。

多队列立项前，应补齐一张小型依赖表：

| 必须记录 | 要回答的问题 |
|---|---|
| WAIT 的队列、地址、宽度、func/ref/mask | 等待哪个版本的标签 |
| 匹配生产者的原始 PM4 包、event/GCR、队列与位置 | 这是怎样的完成保证 |
| 剔除输入最后一次写、输出最后一次读/写 | RAW、WAR、WAW 是否阻止提前 |
| BDA、GDS、图像别名与映射版本 | 描述符之外是否仍有依赖 |
| 实际 `vkQueueSubmit` 时间及 GPU 时间戳 | 1.82 ms 中多少属于录制延迟、依赖等待或设备排队 |

必须在 `CpOpReleaseMem()` **归一化事件参数之前**保存原始字段。仅记录最终 `WriteAtEndOfPipe()` 参数可能丢失客体事件的信息。

资源无冲突也不能自动取消客体显式完成语义。如果该标签确实要求尾部完成，多队列同样必须保留这条边。

### 4. 单队列并不天然意味着命令缓冲逐个完整串行

Vulkan 的提交顺序本身不建立完整的执行与内存依赖；不同命令缓冲之间允许重叠。这里真正值得查的是程序额外建立的依赖。[Vulkan 执行模型](https://docs.vulkan.org/spec/latest/chapters/fundamentals.html)。

当前至少有三处：

- `CompleteDispatch()` 在批次边界加入 `ALL_COMMANDS → ALL_COMMANDS` 依赖；已有启动配置使用 128 dispatch 一批。
- `CommandBuffer::Handle()` 结束计算链时加入 `Compute → ALL_COMMANDS` 依赖。
- `ShaderWriteHazardBarrier()` 使用 `ALL_COMMANDS → Shader` 的广泛依赖。

见 [批次屏障](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/commandScheduler.cpp:115)、[计算链屏障](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/context.cpp:30)、[着色器访问屏障](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.cpp)。

若之前“跳过全部全屏障”的实验只拦截 `EmitGlobalBarrier()`，它没有覆盖上述全部来源，不能作为全部同步优化的上限。

但单队列仍有一个限制：**回读等待的全局 tick 是提交前缀完成点**。即使剔除执行提前，较早提交的无关工作也可能继续拖住结果发布。仅观察剔除 GPU 时间戳提前是不够的。

## 三、按预期收益排序的五条路径

下表是**通过先行实验后的收益预算，不是已测结果**。没有动态依赖和着色器样本支持的方向，实际收益可能为零；各项存在重叠，不能直接相加。

| 排序 | 路径 | 关键路径收益预算 | 工作量估计 | 主要风险 |
|---|---|---:|---|---|
| 1 | 着色器代码质量：wave64、无效 LOD 反馈开销 | 1.0～2.5 ms | 1～3 人周 | 适用覆盖率、wave 语义 |
| 2 | 参数表驱动的跨对象 draw / 独立 dispatch 批处理 | 0.8～2.0 ms | 2～5 人周 | 状态、资源与副作用依赖 |
| 3 | 在生产者附近完成反馈，复用已有快照 | 0.5～1.5 ms | 2～5 人日试点 | 快照版本和页所有权 |
| 4 | 受限的第二队列计算/回读路径 | 0～1.8 ms | 2～4 人周试点 | 标签依赖、跨队列生命周期 |
| 5 | 单队列提交与屏障作用域优化 | 0～0.6 ms | 3～7 人日 | 错误消除真实依赖 |

### 路径 1：优先降低关键 GPU 着色器的执行成本

**机制。** 两个具体切入点已经在代码中确认：

1. wave64 在宿主 subgroup32 上使用 `lane_count=2`，为多数 IR 指令生成上下两个 half，工作组宿主 invocation 数相应减少。这是正确性实现，不等于一定低效，但值得检查寄存器压力、指令量和占用率。[wave64 选择](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/shader/recompiler/backend/spirv/SpirvEmitter.cpp:336)、[双 half 发射](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/shader/recompiler/backend/spirv/spirvEmitterProgram.cpp:276)。

   可先对**证明不依赖完整 wave64 通信和控制语义**的程序生成一客体 invocation 对应一宿主 invocation 的路径。不能直接把 wave64 改成 wave32；EXEC、SGPR、跨 lane 操作、LDS 与同步都要纳入资格判断。

2. 像素着色器 LOD 反馈的 `OpImageQueryLod` 位于 enabled 分支之外。即使该纹理没有启用统计，仍生成查询及相关计算。[EmitLodStats](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/shader/recompiler/backend/spirv/spirvEmitterImage.cpp:13)。

   可以依据实际描述符的统计启用状态生成受控变体。统计启用时保留原语义；已有 subgroup 原子归约不应重复作为新收益。

**收益依据。** 假设后处理 2.3 ms 与帧首剔除 1.6 ms 都有可优化部分，降低 20%～30% 是约 **0.8～1.2 ms GPU 时间**。达到 2～2.5 ms 需要覆盖更多关键计算/绘制，不能只靠后处理链。

**修改位置。** 上述 emitter 文件，以及 `spirvEmitterModule.cpp` 的工作组布局、`pipelineCache.cpp` 的变体键。LOD 变体还必须接入 `native-xpr.inc` 的管线选择：当前 LOD 字段可仅作为数据补丁变化。

**便宜先行实验。** 从关键链选择少量耗时最高的程序，比较原版与一个窄范围、语义完整的优化变体；记录 GPU 时间和输出。LOD 先测试统计全部关闭的合法变体；wave64 先测试最容易证明资格的程序。若可覆盖的关键 GPU 时间不足约 4 ms，就下调该路线的整体预算。

**风险。** 导数查询不能随意移入非一致分支；不能牺牲纹理反馈、浮点语义或跨 lane 正确性。CPU 层面的 PGO/LTO 也不能直接解决这里的 GPU 指令质量。

### 路径 2：从“每对象录制”升级为“参数表批处理”

**机制。** 当前原生 XPR 已经把**同一记录、连续参数**合并成多 draw，但每个记录仍上传常量、绑定描述符集及动态偏移。单纯“改用 multi-draw”已经做过一部分。[NativeXprEmit](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/local/native-xpr.inc:2044)。

新增价值在于：

- 相邻且管线、附件、索引状态兼容的不同对象，共用 GPU 参数表，由 draw 索引选择对象数据。
- 同程序且相互独立的 dispatch，把每次变化的基址、常量、资源地址变成表项，批量同步资源、发射工作。
- 环形分配重定位只更新表中数据；继续保留必要的 SRT 求值和形状校验。

这与失败的“逐 dispatch 原样缓存”不同。不能再次为每个新地址重建整套记录和描述符；也不能把已有 AOT SRT 再编译一遍当作主要收益。

**收益依据。** 3,000 次热剔除 dispatch，若每次减少 0.4～0.7 µs，节省 **1.2～2.1 ms CPU**。其中只有推进 GPU 工作供给的部分转化为帧收益。跨对象 draw 合并和减少 GPU 发射固定成本提供额外空间，合计先按 **0.8～2.0 ms 帧时间**立项。

128 次 dispatch 的整批 GPU 时间不能都算成发射开销：必须先测固定成本占比。

**修改位置。** [DispatchDirect 各阶段](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/renderCompute.cpp:487)、`descriptors.cpp` 的 `FindBuffers/RebindBuffers/CommitBindings`、`native-xpr.inc` 的 `NativeXprEmit`、`renderDraw.cpp` 的准备与发射路径，以及着色器参数表寻址。

**便宜先行实验。** 先只统计相邻工作可合并比例和拒绝原因，再对一小段离线捕获比较：

- 原始 N 次发射；
- 保持输出相同的参数表批处理；
- 辅助性的空工作负载对照，用于估算发射固定成本。

若主要耗时来自着色本身、合并资格很低，或 CPU 节省集中在 GPU 已积压的尾部，就不扩展实现。

**风险。** 必须保持 draw 顺序；dispatch 有 RAW/WAR/WAW、原子、GDS、间接参数或未知 BDA 时回退。不能按 shader 哈希、场景或固定地址认定独立。

### 路径 3：让回读等待生产者的完成点，而非新追加的队尾复制

**机制。** 当前代码有一个具体缺口：

- `ScheduleCopyFeedback()` 可以保留提前录制的反馈快照。
- `TryReadCopyFeedback()` 只在同步 `ReadMemoryOnGpu()` 路径尝试。
- 正常异步路径先进入 `BeginGuestReadback()`，通常重新录制复制并 Flush。
- 现有快照尚未完成时，`TryReadCopyFeedback()` 直接失败，不能将其完成 token 交给异步读者等待。

见 [已有快照机制](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/cache/bufferCache.cpp:309)、[异步入口](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/host_gpu/renderer/cache/bufferCache.cpp:738)。

建议先让异步入口复用**版本有效、完整覆盖所需脏字节**的快照；再考虑等待尚未完成但仍有效的快照，而非在后续无关工作之后追加复制。随后才扩展到通用生产者驱动的反馈。

**收益依据。** 18 次串行回读目前合计约 3.1～7.2 ms，但大量时间可能与渲染重叠，也可能是真实生产者依赖。若其中每次可避免 0.05～0.1 ms 的额外等待，原始空间约 0.9～1.8 ms；帧时间先按 **0.5～1.5 ms**预算。

之前的反馈预取仅修复拆分退化，不能当作本路线已有正收益。这里首先验证的是**当前未拆分基线的快照复用缺口**。

**修改位置。** `BeginGuestReadback`、`TryReadCopyFeedback`、`CopyGuestReadback`、快照槽生命周期，以及 dispatch/copy 生产者的反馈登记。

**便宜先行实验。** 只统计当前异步回读发生时，有多少已有快照满足覆盖及版本要求，并记录：

> 快照完成时间，与实际新回读完成时间之差，是否推进客体后续提交。

若命中率低，或后续提交完全没提前，停止扩展，不再凭“60 次回读”推算收益。

**风险。** 必须处理覆盖后的 GPU 写入、CPU 写入、映射变化、图像别名、槽复用和整页解除保护条件。不能使用上一帧结果，也不能因为命中了某几个字节就把整页交还 CPU。

### 路径 4：采用受限第二队列，而非一开始改造全部资源时间线

**机制。** 先只支持一类资源边界清楚的计算任务：

- 明确输入生产者及客体标签依赖；
- 缓冲资源可固定生命周期，未知 BDA、复杂图像别名先回退；
- 使用独立命令池、上传/回读空间和计算完成 token；
- 在主队列第一次消费、覆盖或回收这些资源之前建立 join。

主队列既有 tick 可继续用于多数资源回收。异步任务持有的资源必须直到 join 完成才进入这套回收规则。这样**不必在试点阶段就把所有 tick 使用者全部转换成双时间线**。

当前设备只创建一个队列，需从 [VulkanCreateDevice](/home/chen-xiao/git/KytyPS5-pr500/_Build/hist/src/src/graphics/presentation/window/vulkanWindow.cpp:570) 开始。优先检查同 family 是否提供第二队列，以降低所有权转移复杂度；仍需显式跨队列同步。

**收益依据。** 保持提交时刻不变：

`3.78 − 1.96 = 1.82 ms`

这是所述帧首剔除排队延迟的理想消除量。实际会受到真实依赖、GPU 资源竞争、回读完成点影响，预算 **0～1.8 ms**。如果标签覆盖整个尾部，相关重叠收益可能为零。

**修改位置。** `graphicContext.h`、`commandScheduler.cpp` 的 `Submit/ReplaySubmit`、`sync.cpp` 的完成事件，以及受限任务对应的 buffer/readback 生命周期。完整通用化还会涉及 `streamBuffer.cpp`、缓存回收及 Native XPR 描述符退休。

**便宜先行实验。** 先建立前述依赖表和理想调度上限，不改队列。只有证明存在足够的合法重叠窗口，再对捕获的小任务使用独立资源验证实际并发。

**风险。** 两个队列不能无序地继续 signal 同一个全局递增 tick；更大的值不能代表另一队列较小值的工作已经完成。跨队列的完成集合需要分别判断，或者通过 join 收敛。[Vulkan 时间线信号规则](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)。

第二队列也不意味着计算吞吐翻倍；上一帧后处理和下一帧剔除可能竞争同一批执行资源。

### 路径 5：先清理单队列中的人为提交依赖

**机制。** 区分两件事：

- 为尽早把工作交给 GPU 而提交；
- 为表达真实依赖而加屏障。

当前 `CompleteDispatch()` 把两者绑定在一起。可以在资源和客体同步证明支持的地方，保留早提交，缩小或消除额外批次依赖；对反馈生产者安排明确的提交边界，避免完成 tick 包含后续无关工作。

这不是针对场景把批次从 128 调成另一个常数，而是让边界由消费者与依赖决定。

**收益依据。** 已测屏障方向收益较小，先按 **0～0.6 ms**。新增空间主要取决于之前的上限实验是否遗漏批次屏障和访问屏障。

**修改位置。** `CommandScheduler::CompleteDispatch`、`CommandBuffer::Handle/ContinueComputeChain`、`ShaderWriteHazardBarrier`、`Buffer::CopyFrom`。

**便宜先行实验。** 分别测量上述屏障类别的覆盖范围与理想上限；危险的省略只用于隔离回放诊断，不作为有效游戏成绩。若上限仍只有约 1%，停止扩大重构。

**风险。** 单队列较细屏障不能取消客体显式同步，也不能让已经提交的命令被任意插队。提前执行但回读完成 token 没提前，也不算成功。

## 四、推荐的组合路径与验收门槛

**实施顺序与收益排序应不同：先做便宜、能迅速证伪的工作，再投入大改造。**

| 阶段 | 工作 | 进入下一阶段的标准 |
|---|---|---|
| 0：校准关键路径 | 补齐标签生产者、资源依赖，以及 CPU/实际提交/GPU/回读统一时间线 | 能解释主提交为何晚到；区分绝对延迟与 flip 原点移动 |
| 1：低风险试点 | 异步回读复用现有快照；LOD 关闭状态的合法变体；确认屏障上限覆盖范围 | 至少一个方向同进程稳定节省约 0.3～0.5 ms，并推进实际完成点 |
| 2：主要收益 | 对合格热程序优化 wave64；参数表批处理先覆盖最有效的相邻工作 | 组合争取进入 **26～27 ms**；渲染 CPU 建议降至约 **21～22 ms** |
| 3：补齐剩余差距 | 在新的基线上重测异步窗口，决定是否实现受限第二队列 | 剩余合法重叠上限足以覆盖离 25 ms 的差距，并有实现开销余量 |
| 4：最终验收 | PGO+LTO，同进程交替至少 4 轮×10 秒；正确性与稳定性验证 | 平均 ≥40 FPS，画面与反馈结果正确，连续游玩 10 分钟无新问题 |

阶段 2 后必须重新计算阶段 3 的空间。**着色器缩短了上一帧尾巴，就可能同时缩小原来的 1.82 ms 排队窗口；两项不能重复记账。** 同理，批处理减少的 CPU 工作和 GPU 发射成本，应在组合运行中确认最终收益。

一个值得争取、但尚未得到实验证实的预算是：

| 不重复记账的关键路径部分 | 目标减少量 |
|---|---:|
| 推进重 GPU 工作供给的 CPU 前缀 | 1.5～2.0 ms |
| 反馈链额外延迟，排除已计入的 GPU 提前量 | 0.8～1.2 ms |
| 其余关键 GPU 工作及排队，着色器与多队列合并计算 | 2.0～2.5 ms |
| **合计** | **4.3～5.7 ms，约 37.9～40 FPS** |

正确性验证应包含计算输出、回读数据及生命周期，而不仅是截图和 Native XPR 记录比较。上一轮图形状态失效问题已经说明：记录内容一致，不等于最终执行状态一致。已有 `ShaderRecompilerComputeTests`、`CompletedCopyFeedback` 和 XPR 离线回放可作为基础。

暂不优先投入扩大写窗口、热页永久免保护、更多逐项 CPU 微优化或再次按 draw/dispatch 拆线程：已有结果不支持足够收益，且免保护会破坏当前 CPU 修改纪元的干净证明。BDA 脏位图减少的是 CPU 查询成本，也没有把保守的 4 GiB 访问范围变成足以支持乱序执行的精确依赖。

**最先值得做的两件事是：查明帧计数标签的生产者与作用域，统计当前异步回读错过有效快照的次数。** 前者决定多队列是否有合法空间，后者提供一个成本较低、可以直接推进客体提交的验证入口。
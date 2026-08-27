# matrix_rank 国产平台问题总结与通用实现改造方案

> 日期：2026-08-27
> 适用分支：`ascend-matrix-rank-triton`
> 讨论对象：`torch.linalg.matrix_rank` 的 FlagGems 通用 Triton 实现及昇腾专用实现

## 1. 文档目的

当前 `matrix_rank` 通用实现已经覆盖 `float32`、`float64`、Hermitian/非
Hermitian、batch、容差和 `out` 等主要语义，但在多个国产平台上暴露出三类
可移植性问题：

1. 依赖跨 program 原子自旋的 software grid barrier，在不同后端出现死锁、
   数据不可见或编译错误；
2. `float32` 输入的 Sturm 计数内部仍无条件使用 `tl.float64`，在无 FP64
   能力的平台上可能编译通过却返回错误结果；
3. correctness test 和 benchmark 使用原生 Torch 作为参考时，可能先触发
   平台原生求解器的未实现接口或不收敛，而 FlagGems kernel 尚未执行。

本文先汇总已经观察到的问题和证据，再简述昇腾实现采用的结构，最后给出一个
不在通用算子中增加 vendor 特判、允许适当性能退化的改造方案。

## 2. 结论摘要

建议把通用实现的默认策略调整为：

```text
k == 1/2
    -> 闭式单 kernel

能安全放入单 program 的小矩阵
    -> fused Jacobi / fused tridiag

Hermitian 中大矩阵
    -> barrier-free Householder tridiagonalization
    -> 依赖 kernel launch 边界完成全局同步
    -> Sturm rank count

非 Hermitian 中大矩阵
    -> barrier-free Golub-Kahan bidiagonalization
    -> 依赖 kernel launch 边界完成全局同步
    -> Sturm rank count

Sturm 精度分派
    -> support_fp64=True：保留当前原生 FP64 高精度路径
       （既可服务 float64 输入，也可作为 float32 输入的内部高精度计算）
    -> support_fp64=False 且输入为 float32：
       走纯 FP32 hi/lo double-single fallback，不出现 tl.float64
    -> support_fp64=False 且输入为 float64：明确报不支持
```

通用算子只根据 `dtype`、`shape`、`hermitian` 和统一能力位
`flag_gems.runtime.device.support_fp64` 选择算法，不出现 `vendor_name == ...`。
第一版优先保证不死锁和结果正确，暂时接受多 kernel launch 带来的性能下降；图捕获
和 panel 优化放到后续阶段，并尽量由 runtime 基础设施提供，而不是写入算子数学
逻辑。

## 3. 当前通用实现的关键风险

### 3.1 Software grid barrier

通用实现的 `_grid_barrier` 使用一个全局原子计数器：

```python
tl.atomic_add(BARRIER, 1)
target = base + (generation + 1) * num_programs
while tl.atomic_add(BARRIER, 0) < target:
    pass
```

它隐含两个强假设：

1. 本次 launch 的所有 program 能同时驻留；
2. 后端能正确生成跨 program、多轮、嵌套循环中的原子轮询和 acquire/release
   可见性。

这两个假设在国产后端上都已被反例打破。

当前 launcher 使用 `chunk` 限制一次处理的 batch 数，但当单个矩阵本身的
`NB` 已经大于设备可驻留 block 数时，`max(1, ...)` 仍会启动完整的单矩阵
grid。例如 Hermitian `k=1024`：

```text
BJ = 8
NB = ceil(1024 / 8) = 128 programs / matrix
```

即使设备只有 16 个 SM，`batch_count=1` 时仍会启动 128 个 program。若已驻留
的 program 在 barrier 自旋并占住资源，尚未调度的 program 永远无法递增计数器，
形成调度死锁。

### 3.2 float32 路径内部依赖原生 FP64

当前 `_matrix_rank_sturm_rank_kernel` 即使输入是 `float32`，也会执行：

```python
d = tl.load(...).to(tl.float64)
e = tl.load(...).to(tl.float64)
atol = tl.load(...).to(tl.float64)
rtol = tl.load(...).to(tl.float64)
```

测试中根据 `runtime.device.support_fp64` 跳过 `float64` 输入，只能保证不会把
FP64 tensor 送入算子，不能阻止 `float32` kernel 内部生成 FP64 指令。因此
“跳过 FP64 用例”与“float32 实现不依赖 FP64”是两个不同问题。

### 3.3 原生 Torch 参考值和 FlagGems 被测路径混在一起

测试辅助函数当前先计算：

```python
native = torch.linalg.matrix_rank(...)
```

之后才调用：

```python
flag_gems.linalg_matrix_rank(...)
```

因此原生求解器未实现、不收敛或卡住时，pytest 会显示停在
`test_linalg_matrix_rank_*`，但并不代表 FlagGems kernel 已经执行。排查时必须
把 native、FlagGems direct 和 `use_gems()` dispatch 放入独立进程分别运行。

## 4. 国产平台问题汇总

| 平台 | vendor | 已观察问题 | 所属层次 | 当前处理/状态 |
| --- | --- | --- | --- | --- |
| 昇腾 | `ascend` | 无可靠跨 program 自旋；同 kernel store/load 不保证顺序；FP64 不可用；部分 BLOCK=64 形态误编译 | Triton 后端与硬件执行模型 | 专用 barrier-free 实现，FP64 fail-fast |
| 海光 BW1000 | `hygon` | Hermitian k=32 FP64 多 program barrier 曾卡死；大尺寸全零矩阵原生 SVD 不收敛 | 通用 kernel 同步；原生 Torch 参考 | barrier 轮询做过战术改写；零矩阵测试改解析期望 |
| 沐曦 C550 | `metax` | 多轮嵌套 grid barrier 稳定卡死；FP64 `tl.dot` 误编译；原生 SVD 对大零矩阵不收敛 | Triton 机器码生成；原生 Torch 参考 | 已有局部 workaround，但 barrier 根因未结构性消除 |
| 平头哥 HGGC | `thead` | 原生批量 Hermitian matrix_rank 调用未实现的 `cusolverDnXsyevBatched_bufferSize` | 原生 Torch/HGGC 参考实现 | correctness 用 CPU reference；benchmark 跳过 batch Hermitian |
| 天数智芯 BI-V150 | `iluvatar` | k=1024 blocked tridiag 128-program grid 在 16 SM 上卡死；缩到 16 program 后完成但 rank=0 | 共驻死锁；无 FP64 平台上的 Sturm 正确性 | 已完成最小二分，等待 barrier-free 和纯 FP32 Sturm 改造 |

### 4.1 昇腾

昇腾实现报告中已经验证：

- 不采用跨 program software grid barrier，同步统一通过 kernel launch 边界；
- MTE3 store 与 MTE2 load 不自动保序，中间 D/E 必须跨 kernel 传递；
- Triton/原生 NPU 求解器均不能提供可用 FP64，FP64 输入在 shape dispatch 前
  明确 fail-fast；
- 某些 `BLOCK=64` 的双对角化形态会因工具链误编译失败，必须改变 kernel 结构；
- Gram 路径存在 `sqrt(eps) * sigma_max` 量级的平方域噪声地板，对缓衰减低秩谱
  可能高估 rank，因此精确路径最终使用线性域 Householder/Golub-Kahan 分解。

昇腾不是通过更换某个 atomic memory order 解决问题，而是从执行结构上取消了
跨 program 自旋。

### 4.2 海光 BW1000

已观察到 Hermitian strict-upper 测试在 `k=32, float64` 进入 tridiag 路径后
卡住：

```text
threshold = 32
herm_tridiag = True
before matrix_rank
<不返回>
```

恢复 `threshold=33` 后该 shape 不进入 tridiag，多次能够正常返回 rank 28。
这个用例中 `BJ=8, NB=4`，规模很小，说明除共驻问题外，后端对多轮原子自旋的
代码生成/可见性也存在风险。

另外，海光原生 Torch 的 SVD 在以下全零矩阵上报告不收敛：

```text
(513, 513)
(1024, 1024)
(2, 513, 513)
dtype = float32 / float64
```

异常发生在 native reference，FlagGems 尚未执行。测试已在提交
`0d4b7ef0b` 中改为解析期望 rank 0，同时保留 FlagGems direct 和 dispatch
验证。

### 4.3 沐曦 C550

沐曦对 `_matrix_rank_herm_tridiag_kernel` 的最小二分证据最完整。复现用例为
`k=32, float64, hermitian=True`，4 个 program 在每列两次、共 62 轮
`_grid_barrier` 中自旋。

已验证现象：

| 结构 | 轮询方式 | 结果 |
| --- | --- | --- |
| 只到达不自旋，外层 62 轮 | `atomic_add(+1)` | 正常，counter=248 |
| 单轮 barrier，多次独立运行 | 多种方式 | 正常 |
| 扁平两轮 | 显式 `acquire` | 卡死 |
| 扁平两轮 | 默认 `acq_rel`、atomic max、volatile 等 | 部分能够正常 |
| 嵌套外层循环两轮及以上 | 所有已测轮询变体 | 卡死 |
| 强制 NB=1 | 无跨 program barrier | 正常返回 rank 28 |

中端 LLIR 中的 atomic、循环位置、线程条件和 block barrier 经核对均符合预期，
问题表现集中在 LLIR 到机器码阶段。`tl.device_print` 放入自旋循环还会独立触发
memory violation，因此最终以 kernel launch 后同步加外部 timeout 判断死活。

提交 `d8a49b2fb` 包含几项有价值的修复：

1. 把 loop-carried atomic 结果改成 while 条件中的直接轮询；
2. FP64 `tl.dot` 误编译时改用逐列 outer-product；
3. V/W workspace 从 `empty` 改成 `zeros`，避免未初始化 NaN/Inf 污染；
4. 原生 MetaX SVD 对大零矩阵不收敛时使用解析期望。

其中第 2～4 项应保留；第 1 项是战术 workaround，不能保证嵌套多轮 barrier
在所有后端可靠，也不能解决 grid 无法共驻。

### 4.4 平头哥 HGGC

批量 Hermitian 输入，例如 `(2, 4, 4)`，原生：

```python
torch.linalg.matrix_rank(aah, hermitian=True)
```

会进入批量对称特征值分解并调用：

```text
cusolverDnXsyevBatched_bufferSize
```

HGGC 报告该 API 不支持。这不是 FlagGems kernel 错误，而是测试 native baseline
在 FlagGems 运行前失败。

当前处理：

- correctness test 中，THead 的 batch Hermitian 原生参考值转到 CPU；
- Hermitian benchmark 中，THead 跳过 batch shape，避免使用 CPU/composed latency
  生成没有可比性的 speedup；
- 二维非 batch Hermitian 和非 Hermitian benchmark 继续保留。

相关提交包括 `70fe2e8ab` 和 `d5a5164e7`。

### 4.5 天数智芯 BI-V150

设备信息和复现结果：

```text
vendor: iluvatar
device: Iluvatar BI-V150
SM count: 16
shape: (1024, 1024)
dtype: float32
hermitian: True
```

独立进程结果：

```text
native Torch                         -> rank 1000，正常
FlagGems blocked, BJ=8, NB=128       -> 180 s timeout
FlagGems unblocked                   -> 完成，但 rank 0
FlagGems blocked, BJ=64, NB=16       -> 完成，但 rank 0
```

由此可以分成两个问题：

1. 将 grid 从 128 降到 16 后不再卡死，强烈证明 blocked 卡死由共驻不足触发；
2. blocked/unblocked 都进入公共 Sturm tail 并返回 0，而 Iluvatar runtime 明确
   声明 `fp64_enabled=False`，当前公共 Sturm 却无条件使用 `tl.float64`，因此
   无 FP64 的 Sturm 路径是 rank 0 的第一嫌疑。

第二点仍建议通过在 tridiag kernel 后打印 `diag/offdiag` 最终确认：若 D/E 正确
且直接按对角计数为 1000，而 Sturm 输出为 0，就完成了最终归因。

## 5. 跨平台共性根因

上述问题可以归并为四个共性根因。

### 5.1 共驻是假设，不是保证

只要 kernel 内存在全 grid 自旋 barrier，就必须证明整个 grid 同时驻留。SM 数
只能给出粗略上限，寄存器、shared memory、warp 数和厂商调度策略都会改变实际
occupancy。`max(1, sm_count // nb)` 无法处理单矩阵 `nb > capacity`。

### 5.2 原子语义在源码、IR 和机器码之间可能失真

即便 LLIR 正确，后端仍可能把循环中的 atomic poll 提出、缓存或降低成缺少可见性
保证的 load。更换 `acquire`、`acq_rel`、volatile 或 loop 写法只能解决部分编译
形态，不能形成跨后端契约。

### 5.3 输入 dtype 能力与内部计算 dtype 是两件事

跳过 float64 输入不能保证 float32 kernel 不生成 FP64 指令。在
`support_fp64=False` 的设备上，float32 fallback 必须从代码结构上避免
`tl.float64`；支持 FP64 的设备则没有必要被迫放弃现有原生 FP64 高精度路径。

### 5.4 测试 oracle 也具有平台能力边界

原生 Torch 可能缺少 batched eigensolver，或某个 vendor SVD 对重复奇异值不收敛。
测试应区分：

- FlagGems 算法正确性；
- FlagGems 注册和 dispatch；
- 平台原生 Torch 是否能作为设备侧 oracle；
- benchmark 是否存在公平的 native baseline。

## 6. 昇腾实现思路简述

### 6.1 不调用原生分解算子

昇腾最终实现的 rank 计算全部由本文件 Triton kernel 完成，不使用
`torch.linalg.svd/eigh/matrix_rank` 作为运行时兜底。原生 Torch 只存在于测试
参考侧。

### 6.2 小矩阵单 program 融合

`k <= 32` 等可安全驻留的形态使用单 program fused kernel，把分解和计数尽量
放在寄存器中。单 program 不需要 grid barrier，只有 block/program 内同步。

### 6.3 Hermitian 大矩阵每步三 kernel

Hermitian 大矩阵采用单边 Householder 三对角化，每一步拆成：

```text
step kernel
    -> 计算当前 Householder reflector、tau、D/E

mat kernel
    -> 多 program 计算对称矩阵向量积/原子累加

apply kernel
    -> 多 program 完成 rank-2 trailing update
```

三个 kernel 在同一 stream 顺序提交，kernel 边界承担全局同步。program 可以分波次
调度，不要求同时驻留，也不使用 atomic 自旋等待其他 program。

### 6.4 非 Hermitian 大矩阵每步六 kernel

Golub-Kahan 双对角化分别执行左、右 Householder：

```text
left step -> left mat -> left apply
right step -> right mat -> right apply
```

每个阶段通过 launch 边界排序。最终 D/E 再由独立 kernel 转换为 Sturm 所需的
三对角形式。

### 6.5 两阶段 Sturm 与纯 FP32 double-single

昇腾大路径把 Sturm 分为：

1. FP32 bracket/bisection 阶段，快速确定阈值或判断是否需要精化；
2. decisive final 阶段，使用 FP32 hi/lo double-single 完成 `+tol/-tol` 的严格
   符号计数。

关键辅助原语 `_df64_add`、`_df64_mul_ds`、`_df64_div_ds` 全部只使用 FP32。
因此可以在没有原生 FP64 的设备上获得比普通 FP32 recurrence 更稳定的结果。

### 6.6 Graph 只用于摊薄 launch 开销

大矩阵可能产生 O(k) 级 kernel launch。昇腾用按 shape/device/stream 缓存的
NPUGraph 捕获整条序列，降低重复执行时的 host enqueue 开销。Graph 是性能优化，
不是正确性前提；关闭 graph 后算法仍应正确，只是更慢。

## 7. 通用实现的改造原则

1. 通用算子中不增加 vendor 判断；
2. 默认 dispatch 不再依赖跨 program software grid barrier；
3. 根据 `flag_gems.runtime.device.support_fp64` 做能力分派，而不是根据 vendor
   名称分派；无 FP64 能力时，float32 fallback 内部不使用原生 FP64；
4. 只使用 kernel launch 边界作为跨 program 的全局排序点；
5. 同一 stream 连续 launch 即可，不在每步调用 host synchronize；
6. 小矩阵继续融合，中大矩阵优先选择结构简单、可验证的 barrier-free 路径；
7. 图捕获、fast launch 和 workspace cache 后置，且尽量放入 runtime 抽象；
8. 不以跳过 correctness case 代替生产路径修复。

## 8. 目标通用架构

### 8.1 Dispatch

建议第一版采用以下统一 dispatch：

```text
empty                 -> zero output kernel
k == 1                -> rank1 closed form
k == 2                -> rank2 closed form
small fused capacity  -> single-program fused path
hermitian=True        -> barrier-free tridiagonalization + Sturm
hermitian=False       -> barrier-free bidiagonalization + Sturm
```

原 blocked Jacobi、blocked tridiag、unblocked tridiag 和 bidiag 中依赖
`_grid_barrier/_neighbor_sync` 的默认分支逐步退出 dispatch。代码可暂时保留用于
对比，但不能继续作为默认 portable 路径。

### 8.2 Barrier-free Hermitian 分解

可以直接复用当前 blocked WY 数学结构和 workspace，把一个大 kernel 拆成多个
阶段，而不必机械复制昇腾代码。

每列或每 panel 的建议阶段：

1. `tridiag_step_kernel`
   - 读取当前列；
   - 计算 sigma、alpha、tau；
   - 写 reflector、D/E；
   - 清零本步 accumulation workspace。
2. `tridiag_mat_kernel`
   - 多 program 覆盖 trailing matrix；
   - 计算 `A v` 和必要修正项；
   - 使用 atomic add 只做有限生命周期的归约，不在 kernel 内等待。
3. `tridiag_apply_kernel`
   - 在前一 kernel 完成后读取完整 accumulation；
   - 执行 `A -= v w^T + w v^T`。
4. 可选 `panel_update_kernel`
   - panel 化成熟后，用 BLAS3/WY 降低 launch 数和内存流量。

第一版可以先使用每列三 kernel 的 unblocked 结构，避免同时引入 panel 数学和同步
重构两类风险。

### 8.3 Barrier-free 非 Hermitian 分解

将当前 Golub-Kahan 的四个 barrier phase 拆成独立 launch，或按昇腾结构拆为
六个逻辑阶段：

```text
left reflector
left matrix-vector reduction
left trailing update
right reflector
right matrix-vector reduction
right trailing update
```

所有中间 workspace 在 launch 前一次分配并循环复用。每一步只 enqueue kernel，
不做 host 同步。

### 8.4 Sturm 能力分派

在 host launcher 层读取 `flag_gems.runtime.device.support_fp64`，选择独立 kernel
或独立 JIT specialization，不把该能力位作为设备侧动态条件塞进同一个 kernel：

```text
输入 float64
    support_fp64=True  -> 当前原生 FP64 Sturm
    support_fp64=False -> 入口处明确报不支持

输入 float32
    support_fp64=True  -> 保留当前原生 FP64 辅助的 Sturm
    support_fp64=False -> 纯 FP32 hi/lo double-single Sturm
```

其中无 FP64 能力设备使用的纯 FP32 double-single fallback 满足：

- D/E、atol/rtol、Gershgorin bound 使用 FP32；
- `e^2`、pivot、除法余项使用 `(hi, lo)`；
- count 接口接受 `(xh, xl)`，不再通过 `tl.float64` 拆分 x；
- 正负阈值使用两条 recurrence 同步推进，减少 launch 和重复 load；
- 对精确 tie 保留当前 strict threshold 语义；
- bisection 的 pad 使用 FP32 可表示的 ULP/epsilon，不使用在 FP32 中会舍入成 1
  的 `1 + 1e-9` 或下溢为 0 的 `1e-292`。

这样只有不支持 FP64 的设备承担 double-single 的额外指令开销；NVIDIA 等支持
FP64 的设备保持当前实现及其性能特征。两条路径通过统一 capability 选择，仍然
不含 MetaX、Hygon、Iluvatar 等 vendor 特判。

### 8.5 float64 输入约束

float64 输入只在 `flag_gems.runtime.device.support_fp64=True` 时进入当前原生
FP64/高精度路径。能力为 False 时应在 launch 前给出稳定、明确的 unsupported
错误；相应 correctness/benchmark 用例根据同一个 runtime capability 显式跳过，
不能依靠编译失败、超时或错误 rank 被动暴露能力不足。

## 9. 分阶段实施计划

### 阶段 0：建立可观测性

- 为 direct/native/dispatch 准备独立 timeout 复现脚本；
- 提供可选 debug 开关，在 tridiag/bidiag 与 Sturm 之间导出 D/E；
- 记录每条路径的 grid、program 数、SM 数、dtype 和 kernel 名；
- debug 默认关闭，不进入性能路径。

### 阶段 1：Sturm 能力分派与无 FP64 fallback

目标：先解决 BI-V150 在解除死锁后返回 rank 0 的问题。

1. 从昇腾实现提取纯 FP32 double-single 原语和 decisive count；
2. 在通用代码中实现仅供 `support_fp64=False` 使用的 float32 Sturm；
3. 在 host launcher 中按统一 capability 选择现有原生 FP64 路径或新 fallback，
   并保证支持 FP64 的平台保持原 dispatch；
4. 保留 factorization 不变，先用 BI-V150 的 `BJ=64, NB=16` 调试配置验证
   `diag/offdiag -> rank 1000`；
5. 跑近阈值、零 pivot、正负特征值、低秩和缓衰减谱测试；
6. 检查编译 IR，确认无 FP64 fallback specialization 不含 f64 类型。

### 阶段 2：Hermitian barrier-free 路径

目标：从结构上解决 MetaX/Hygon 的多轮 atomic spin 和 Iluvatar 的共驻死锁。

1. 实现 step/mat/apply 三 kernel；
2. 首先覆盖 `k >= 256`，再向 33～255 扩展；
3. 保留只读下三角语义；
4. 用 kernel 边界传递 D/E、reflector 和 accumulation；
5. 默认 dispatch 切到新路径；
6. 删除 launcher 对 `_sm_count` 和 resident block 数的正确性依赖。

### 阶段 3：非 Hermitian barrier-free 路径

1. 拆分 Golub-Kahan 左右阶段；
2. 覆盖 513、1024、长宽矩阵及 batch；
3. 将中等尺寸 blocked Jacobi 默认 dispatch 逐步迁移到 barrier-free Householder；
4. 停用 `_neighbor_sync` 和 Jacobi grid barrier。

### 阶段 4：清理和性能回收

1. 确认默认路径不再调用 `_grid_barrier/_neighbor_sync`；
2. 删除失去入口的旧 kernel，或用明确实验开关隔离；
3. 评估 panel/WY 合并，减少每列 launch；
4. 若 runtime 能提供统一 graph capture，再按 shape/device/stream 捕获；
5. workspace cache 必须处理并发、stream 隔离、设备隔离和淘汰时同步。

## 10. 测试方案

### 10.1 死锁回归

每个平台使用独立进程和外部 timeout，至少覆盖：

```text
k = 32, 33, 64, 65
k = 128, 255, 256, 257
k = 512, 513, 1024
batch = 1, 2, 4
hermitian = False / True
```

重点复现：

- MetaX/Hygon：`k=32, float64, hermitian=True`；
- Iluvatar：`k=1024, float32, hermitian=True`；
- 单矩阵 program 数大于 SM 数的情况；
- 多 batch 下 grid 分波次调度的情况。

### 10.2 正确性

- 对角、单位、全零、缺秩；
- 随机稠密低秩；
- 缓衰减谱和近阈值谱；
- `rank(A) == rank(A.mH)`；
- `AA^H` 的 SVD 与 Hermitian 路径一致；
- Hermitian 只读下三角；
- 默认/atol/rtol/二者同时传入；
- scalar、0D tensor、per-batch 和广播容差；
- output shape、dtype、device 和 `out` 语义；
- `support_fp64=False` 时选中的 float32 fallback IR 不含 f64；
- `support_fp64=True` 时 float32 保持现有原生 FP64 辅助路径；
- float64 只在 runtime 宣称支持时运行。

对于原生 Torch 不可用的参考场景：

- 数学期望明确时使用解析值；
- correctness 可使用 CPU native reference；
- benchmark 没有公平 device baseline 时跳过对应 shape，不报告伪 speedup；
- 不跳过 FlagGems direct/dispatch 本身。

### 10.3 性能

第一阶段只设置宽松回归门槛，记录而不阻塞正确性合入：

- kernel launch 数；
- 第一次编译/捕获时间；
- 热启动 latency；
- workspace 峰值；
- k=256/512/1024 的性能下降比例；
- batch 对吞吐的影响。

## 11. 验收标准

1. 通用 `src/flag_gems/ops/linalg_matrix_rank.py` 不出现 vendor name 判断；
2. 默认 dispatch 不调用跨 program 自旋 barrier；
3. `support_fp64=False` 时，float32 fallback specialization 不生成
   `tl.float64`；支持 FP64 的设备保持原生 FP64 路径；
4. BI-V150 `k=1024` 不超时且返回 rank 1000；
5. MetaX/Hygon 原卡死用例在 timeout 内稳定重复通过；
6. NVIDIA 等原有平台完整 correctness 不回退；
7. 所有中间 buffer 在读取前有明确 producer 和 kernel-boundary ordering；
8. 原生 Torch baseline 失败不会被误报为 FlagGems kernel 失败；
9. 性能下降有 benchmark 数据和后续优化记录，但不以恢复 software grid barrier
   作为默认优化手段。

## 12. 风险与取舍

### 12.1 Launch 开销

每列三到六个 kernel 会增加 host enqueue 和设备调度开销。第一版接受该成本，后续
通过 panel 化或 runtime graph capture 回收；不能为了性能重新引入没有跨后端保证
的 grid 自旋。

### 12.2 Atomic reduction

barrier-free 路径仍可在单个 mat kernel 内使用 atomic add 汇总 partial，但该
kernel 内不读取最终结果、不自旋等待。下一个 kernel 在 launch 边界后读取，语义
明显强于同 kernel producer/consumer。

### 12.3 double-single 精度

纯 FP32 double-single 必须针对零 pivot、subnormal、Inf/NaN、严格阈值和缓衰减谱
做专项验证。不能只凭对角输入通过就替换所有路径。

### 12.4 Workspace 与并发

多阶段算法需要更多 workspace。第一版每次调用独立分配最简单；若加入缓存，必须
把 device、stream、shape、dtype 和 batch 纳入 key，并处理并发复用和淘汰。

## 13. 建议的近期动作

1. 在 BI-V150 上先打印 blocked `BJ=64, NB=16` 后的 D/E，最终确认 rank 0 位于
   Sturm tail；
2. 把昇腾 `_mr_sturm_big_tridiag_kernel` 和
   `_mr_sturm_final_tridiag_kernel` 的纯 FP32 double-single 思想移入通用代码，
   作为 `support_fp64=False` 时的 fallback；
3. 在 host launcher 使用 `flag_gems.runtime.device.support_fp64` 分派，支持 FP64
   的平台保留现有 Sturm，不加 Iluvatar 特判；
4. 随后实现 Hermitian step/mat/apply 三 kernel，优先覆盖 k=1024；
5. 在 MetaX、Hygon、Iluvatar、NVIDIA 四个平台跑同一组 timeout correctness；
6. 确认稳定后再推进非 Hermitian bidiag 和 blocked Jacobi 的 barrier-free 改造。

该路线的核心不是为每个国产后端继续增加补丁，而是把通用算法建立在所有 GPU
后端都能提供的最小契约上：单 kernel 内局部同步、kernel launch 边界全局排序、
FP32 基础算术和显式 workspace 数据流。

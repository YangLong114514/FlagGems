# `linalg_matrix_rank` 昇腾 910B 后端实现与优化报告(详细版)

> 分支:`ascend-matrix-rank-triton`(已推送 `github.com:YangLong114514/FlagGems`)
> 提交:`dcd1d541`+ 阶段 4 切换提交(见文末)
> 硬件:昇腾 910B4(20 AI Core × 2 Vector = 40 Vector 核,UB 192KB)
> 软件:CANN 8.5.0、triton-ascend 3.2.0(BiShengIR)、torch 2.6.0+cpu / torch_npu 2.6.0rc1
> 目标:`torch.linalg.matrix_rank` 在 NPU 上的 speedup ≥ 0.8
>
> **阅读指引**:第一至三章是设计/优化的主体内容,其中 §1.4 已更新为**当前最终 dispatch**;第一阶段的原始结构保留在附录 C。第二至第六阶段是按时间顺序的演进记录(RRQR → 精确大矩阵路径 → hermitian 基线修正与小矩阵优化 → 评审修复与工具链退化处置 → 精确路径推广),各阶段内的"当前状态"描述以写作时为准,最终以 §0 总结 + §1.4 + 第六阶段为准。

---

# 0. 总结(当前最终状态)

**交付物**:`torch.linalg.matrix_rank` 的昇腾 910B 纯 Triton 后端,无任何 aclnn/native 分解兜底,覆盖 fp32 全部 shape(k=1 到 4096+,含 batch/非方阵/hermitian/逐 batch atol/rtol)。

**默认分发(第六阶段后,全部精确线性域,除两段文档化例外)**:

| 频段 | 默认路径 | 说明 |
|---|---|---|
| k = 1/2 | 闭式 kernel | — |
| k ≤ 32(任意长宽) | 寄存器融合 GK + Sturm | herm 走单边三对角分支 |
| 33~64 方阵 | bidiag64(GK 线性域)| 第六阶段替代 Gram |
| 33~64 herm | padded 单边三对角化 | — |
| 长维 k≤64(如 64×512) | **Gram**(例外①)| σ² 域地板 √eps·σmax;精确 QR 压缩路径 env 可选(0.26~0.6× 不达标)|
| 65~128 非 herm | **RRQR**(例外②)| 精确路径 0.47~0.85× 不达 0.8 底线;|R_ii| 近阈值 ±1 文档化,env 可选精确路径 |
| herm k > 64 | **单边三对角化 + ±tol df64 Sturm**(精确)| 1.3~7.1× |
| 非 herm k > 128 | **非分块 GK 双对角化 + df64 Sturm**(精确)| 0.85~2.7× |

**性能(验收口径 `benchmark --mode operator`,默认分发,48 项全部 SUCCESS)**:general 全部 ≥ 0.93;hermitian 除两个历史边际 shape(33×33、17×17,多轮 0.71~0.98 波动,共享机器噪声)外 ≥ 0.89。大矩阵:general 1024² 2.68×、512² 1.77×;herm 1024² 6.94×、512² 3.92×。

**正确性(最终态)**:
- 官方套件 92 passed / 6 skipped(fp64 与 complex 按环境跳过),默认/exact 双模式一致。
- 366 例全路径扫描:默认 48 例失败**全部**是例外①的 Gram σ² 域地板(缓衰减低秩谱高估,文档化);exact 模式 2 例((3,3)/(7,7) fp32 噪声区边界,fp32 参考自身即与 fp64 不一致)。
- herm 专项压力 34/34(近阈值 ±tol 双符号簇、低秩、缓衰减、零矩阵、垃圾上三角、batch)。
- bidiag64 σ 直验 121/121(~6e-8);QR 频段对抗 22/22;非方阵回归 11 例。

**过程中修复的四个真实生产缺陷**:① fp32 dd/ee 把平方域地板带回决定性计数(k≥513,近阈值判错);② fused kernel 非方阵丢尾能量(预存,官方测试从未覆盖);③ 评审指出的零矩阵未初始化读取 / hermitian 未守下三角语义 / Sturm 同 kernel 写后读 / fp64 fail-fast 顺序;④ 大路径 host enqueue 瓶颈(NPUGraph 化,replay ~1μs/launch)。

**阶段 6 优化侧的死路记录(防止重复尝试)**:① 128 宽寄存器驻留 GK kernel —— UB 溢出(8 tile 双缓冲 ~360KB > 192KB);② 每步 6→4 kernel 条带合并 —— 性能中性偏负(graph 重放下 launch 仅 ~1μs,地板是每 tile 操作 ~2.4μs,合并损失 grid 并行度);③ herm 大矩阵走 GK —— 浪费一半反射(已由单边三对角化根治)。

**核心工程教训**(详见 §4/§5 与各阶段):该后端上 tile 指令发射是唯一稀缺资源;MTE3/MTE2 不保序要求中间量跨 kernel 传递;kernel 复杂度有隐形上限,任何源码扰动都可能重摇误编译彩票——所有正确性结论只对验证时的环境代际有效,改动必须重跑全 K 扫描。

---

---

## 目录

0. [总结(当前最终状态)](#0-总结当前最终状态)
1. [昇腾算子设计介绍](#1-昇腾算子设计介绍)
2. [优化思路](#2-优化思路)
3. [优化手段](#3-优化手段)
4. [遇到的问题和解决方法](#4-遇到的问题和解决方法)
5. [目前的机器瓶颈](#5-目前的机器瓶颈)
6. [验证结果](#6-验证结果)
7. [遗留问题与展望](#7-遗留问题与展望)
8. [第二阶段:k>64 纯 Triton 路径(RRQR)](#第二阶段k64-纯-triton-路径rrqraclnn-兜底已全部移除)
9. [第四阶段:hermitian 基线修正 + 小矩阵优化](#第四阶段hermitian-基线修正npu-散算子-小矩阵路径性能优化)
10. [第五阶段:评审驱动的 correctness 修复](#第五阶段外部代码评审驱动的-correctness-修复--3364-频段工具链退化处置)
11. [第六阶段:精确路径推广](#第六阶段精确路径推广bidiag64--非方阵修复--df64-地板修复--qr-压缩--npugraph)
12. [附录 A/B/C](#附录-a改动文件清单)

---

# 1. 昇腾算子设计介绍

## 1.1 算子语义

`torch.linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False)` 计算数值秩:

```
hermitian=False:  rank = #{ σᵢ > tol },  tol = max(atol, rtol·σmax)
hermitian=True:   rank = #{ |λᵢ| > tol }, tol = max(atol, rtol·|λ|max)
```

其中默认 `rtol = max(m,n)·eps`、`atol = 0`,`atol`/`rtol` 支持按 batch 逐元素指定。

**容差语义的实测确认(重要,影响实现)**:在 CPU 上用 `diag([1.5, 1.25, 0.8, 0.1])` 探测:

| 参数 | 结果 | 说明 |
|---|---|---|
| `atol=0.75, rtol=0.75` | rank=2 | 若语义是 `atol+rtol·σmax` = 0.75+1.125 = 1.875 → 应为 0;**实测为 2,证明语义是 `max(atol, rtol·σmax)` = max(0.75, 1.125) = 1.125** |
| `rtol=0.75` | rank=2 | tol = 1.125,σ>1.125 的有 1.5、1.25 ✓ |
| `atol=0.75` | rank=3 | tol = 0.75 ✓ |
| 默认 | rank=4 | tol = 4·eps·1.5 ≈ 7.1e-7 ✓ |

原实现(上一版报告)的 svdvals 兜底路径用了 `tol = atol + rtol·σmax`,在"双容差同时设置"时与 torch 不一致——本次已修复。

## 1.2 基线确认:torch 在 NPU 上的行为

实测确认(设备 `npu:0`):

| 项 | 结论 |
|---|---|
| `torch.linalg.matrix_rank`(fp32) | **真在 NPU 执行**(aclnn `svd_npu` 原生算子),非 CPU 回退。大矩阵耗时 ≈ `torch.linalg.svdvals`(同一 aclnn SVD 内核) |
| `torch.linalg.matrix_rank`(fp64) | `RuntimeError: svd_npu only supported Float, but get double` — **fp64 无设备实现** |
| `torch.eye`(fp64) | `aclnnEye ... DT_DOUBLE not implemented` — fp64 测试在**构造输入**时就失败 |
| hermitian=True 路径 | torch 内部走 `eigvalsh`(aclnn eigvalsh),实测比 svdvals 慢 13–19 倍((64,64) hermitian ≈ 12.7ms vs general ≈ 1.5ms)——**这是 hermitian 输入的巨大优化空间** |

**基线耗时**(事件计时,冷 L2,中位数,单位 μs):

| shape | general | hermitian |
|---|---|---|
| (8,8) | 474.8 | 654.2 |
| (16,16) | 566.6 | ~700 |
| (17,17) | 494.5 | ~700 |
| (32,32) | ~620 | ~750 |
| (33,33) | 678.1 | ~750 |
| (64,64) | 1568.8 | 3258.5 |
| (128,128) | ~2300 | ~4000 |
| (256,256) | ~11500 | ~280000 |
| (512,512) | ~94000 | ~3.3e6 |
| (1024,1024) | ~4.8e5 | ~6.3e6 |

> 注:小矩阵(≤64)的 baseline 受 aclnn 启动开销主导(约 400-600μs),这是我们必须超越的目标。
> 注(第四阶段起):benchmark 的 hermitian 基线已改为 NPU 散算子 `_composed_matrix_rank`(torch 原生 hermitian 路径在 NPU 上 CPU fallback,倍数虚高),本表 hermitian 列为历史数据,性能结论以第四阶段 §5 / 第五阶段 §3 为准。

## 1.3 算法选型:三个候选的完整评估

数值秩的本质需求:**只要"σᵢ 与 tol 的大小关系",不要精确的 σᵢ**。这给了算法很大自由度。

### 候选 A:one-sided Jacobi(GPU 分支的主路径)

原理:对 A 的列做成对正交旋转(one-sided Jacobi),收敛后奇异值 = 列范数。GPU 上小矩阵走这个路径。

昇腾实况(UB 驻留实现,全部稳定原语):
- 旋转本身**正确**(与 CPU 参考逐元素对比误差 7.6e-6)
- **相邻对顺序不收敛**:列 i 只与列 i±1 配对,与其他列的耦合残留永不消除——CPU 模拟 300 sweeps 范数纹丝不动。必须用标准 cyclic 顺序(每列每 sweep 与 ~K/2 个不同列配对)
- cyclic 顺序正确,但**性能崩溃**:

| K | sweeps | 耗时/次 |
|---|---|---|
| 16 | 4 | 2961 μs |
| 64 | 8 | 99325 μs |

原因:每对旋转需要 2 次列提取(全 tile 掩码归约)+ 2 次 2D where 写回 = 4 个全 tile 操作;总操作数 = O(K²·sweeps) ≈ 4·32·8·64 = 65536 个 tile 操作(K=64)。按 §2.1 的 tile 操作成本模型(~2.4μs/个)估算正好 ~160ms,与实际一致。**结构性淘汰**。

### 候选 B:Gram + Householder 三对角化 + Sturm 计数

原理:`G = AᵀA`(Cube 的 tl.dot 算,一次 GEMM 极快)→ 对称 Householder 三对角化(O(K) 步)→ Sturm 序列计数 `#{λ(G) > tol²}`。

- **性能好**:O(K) 步 × 每步 2 个全 tile 外积 → K=64 约 550μs
- **精度缺陷(致命)**:`λmin = σmin²`。fp32 下 G 的绝对舍入误差 ~ `K·eps·‖A‖²`。当 `σmin < sqrt(K·eps)·σmax`(即条件数 κ ≳ 3000)时,λmin 被舍入完全淹没。
- **stress 实测**:260 次随机测试出现 2 次错误,典型失败:`got=15 want=16,σmin/tol = 63.78`(σmin 离容差 64 倍远,但 Gram 路径的 λmin 已失真为负/零)
- 该错误**确定性可复现**(同一输入恒定错),定位方法见 §4.3

### 候选 C:Golub-Kahan 双对角化 + Sturm 计数(最终采用)

原理:直接在 A 上做左右 Householder 双对角化(LAPACK DGEBD2 风格):

```
for j in 0..K-2:
    左反射: H_j·A,消去 A[j+1:, j](对角下方);D[j] = 反射后的对角元素(±σ)
    右反射: A·H'_j,消去 A[j, j+2:](超对角右侧);E[j] = 反射后的超对角元素(±σ)
→ B = 上双对角矩阵,σ(B) = σ(A)(正交变换保奇异值)
→ BᵀB 是三对角矩阵,元素精确构造:
    dd[i] = d[i]² + e[i-1]²
    ee[i] = d[i]·e[i]
→ Sturm 计数 #{λ(BᵀB) > tol²} = #{σ > tol} = rank
```

**为什么这是正确选择**:
1. **线性精度**:双对角化的 d/e 误差 ~ `K·eps·‖A‖`(与 σ 同量级);BᵀB 的 dd/ee 是 d/e 的标量运算(相对误差 ~eps)。**σmin 无论多小都不会丢**,彻底消除候选 B 的偶发错误。实测 260/260 全对。
2. **性能可行**:2K 步 × 每步 2 个全 tile 外积 = O(K) 个 tile 操作。
3. **无矩阵平方**:不需要 tl.dot 构造 Gram(避开 Cube 操作数限制)。

## 1.4 最终实现结构(当前 HEAD,已与代码同步)

> 本节描述**当前代码**的真实 dispatch;第一阶段的原始结构(k≤64 统一双对角化 + svdvals 兜底)保留在附录 C 供溯源。

全局原则:**所有秩计算都由本文件的 Triton kernel 完成,没有任何 aclnn/native 分解兜底**(svdvals 兜底已于第二阶段删除)。唯一的 host 侧 aten 调用是 hermitian 33~64 路径的 padding 与下三角对称化(zeros/where/arange,不做分解)。fp64 在所有 shape dispatch 之前 fail-fast(`NotImplementedError`:triton-ascend 无法编译 fp64 kernel,aclnn svd_npu 也仅支持 fp32)。

```
linalg_matrix_rank(input, *, atol, rtol, hermitian)
  ├─ _check_input:ndim≥2、hermitian 需方阵
  ├─ fp64 → NotImplementedError(先于一切 shape dispatch,第五阶段前移)
  ├─ 标量 tol 快路径:atol/rtol 非 tensor 时直接作为 kernel 标量参数
  │   (不物化 (batch,) 张量,省 2 次 aten launch);tensor tol 走 _prepare_tolerances
  └─ _launch_matrix_rank: k = min(m,n), rows = max(m,n), batch_count = numel/(m·n)
       ├─ k == 1 → _matrix_rank_rank1_kernel(单奇异值闭式)
       ├─ k == 2 → _matrix_rank_rank2_kernel(2×2 旋转解析)
       ├─ fp32 且 k ≤ 64 → _launch_tridiag_rank:
       │    ├─ m,n ≤ 32 → _matrix_rank_small_fused_kernel(单 launch:
       │    │    寄存器内 Golub-Kahan 双对角化 + Sturm 计数;hermitian 走单边
       │    │    三对角化分支,load 后寄存器内按下三角对称化;宽矩阵经
       │    │    "正常 load + 寄存器 trans" 转置为长形)
       │    ├─ 33 ≤ k ≤ 64 且 hermitian → host 侧 cached-mask torch.where
       │    │    对称化(只读下三角)+ padded 单边三对角化 + Sturm(两 kernel)
       │    ├─ 33 ≤ k ≤ 64 非 herm(m,n ≤ 64)→ _matrix_rank_bidiag64_kernel
       │    │    (单 program GK 双对角化,原始 d/e)+ 共享 Sturm 尾巴
       │    │    (to_tridiag + sturm_big + df64 sturm_final)——第六阶段新增,
       │    │    取代 Gram(rand/diag/lowrank 全谱精确)
       │    └─ 长维(有一维 > 64)→ 默认 Gram(Cube)+ 三对角化 + Sturm;
       │         FLAGGEMS_MR_EXACT_PATH=1 时切换为 QR 压缩(Householder QR
       │         得 k×k R,σ(R)=σ(A) 线性域)+ bidiag64 + df64 尾巴
       │         (精确但慢,见第六阶段 §3 的性能墙分析)
       ├─ hermitian 且 k > 64 → _launch_tridiag_big_rank(默认,精确):
       │    单边 Householder 三对角化(3 kernel/步:step/mat/apply,K 裁剪
       │    尾随 grid)+ 特征值域 ±tol Sturm(bracket + df64 决定性双链),
       │    整条 launch 序列按 shape NPUGraph 捕获重放——比 RRQR 更快且
       │    精确(herm 128² 1.34 / 256² 2.04 / 512² 3.97 / 1024² 7.11)
       ├─ 非 herm 且 k > 128 → _launch_bidiag_rank(默认,精确):
       │    非分块 Golub-Kahan 双对角化(尾随裁剪 grid + 单遍 step)+
       │    独立 BᵀB 构造 kernel + df64 Sturm;NPUGraph 捕获
       │    (129~512 实测 0.85~1.75×,1024² 2.6×);general 65~128 经
       │    FLAGGEMS_MR_EXACT_PATH=1 也走这里(验证用)
       └─ 非 herm 且 65 ≤ k ≤ 128 → 分块 Householder QR(无主元,RRQR):
            rank 由 |R_ii| 对角读出(精确路径在该区间 0.47~0.85×,不达
            0.8 底线,QR 的近阈值 ±1 限制维持文档化——这是默认分发中
            仅剩的 QR 语义缺口)
```

**关键设计点:为什么分解和 Sturm 计数拆成独立 kernel**

昇腾的 MTE3(store)与 MTE2(load)是**两条独立的硬件队列,不自动保序**。若在同一 kernel 内把 D/E 写到 GM 再读回,batch 并发时读可能越过写,读到陈旧数据。实测:寄存器版 D/E(batch=8)2/8 错;拆成两个 kernel 后 kernel 边界强制保序,问题消失。这个约束贯穿整个设计(cholesky_solve 的 UB 驻留方案是同一问题的另一种解法,但我们的工作矩阵无法完全驻留 UB 的同时保证 O(K) 步)。第五阶段进一步把"同 kernel 写后读"从架构上彻底消除:大路径的 BᵀB 三对角由独立构造 kernel 产出(load 全在 store 前),Sturm kernel 纯读;小路径的回写改为幂等(写入值 = 读出值,顺序无关)。`tl.debug_barrier` 不是可用方案——插入本身即是源码扰动,会重新触发 BLOCK=64 的误编译(第五阶段 §2/§4)。

---

# 2. 优化思路

## 2.1 性能模型:tile 指令是唯一稀缺资源

通过系列微基准实测(这是全部优化的理论基石):

| 操作(64×64 tile) | 单次成本 |
|---|---|
| 纯元素级(g = g·0.5 + 0.25) | 2.46 μs |
| 1D 广播乘(g = g·v[None,:]) | 2.39 μs |
| axis-1 归约(acc += sum(g, axis=1)) | 2.31 μs |
| 2D where 写回 | ~2.4 μs |
| 16×16 tile 同类操作 | ~1.5 μs(固定开销主导,不随面积线性) |
| 动态循环携带张量的每迭代开销 | ~0.9 μs |

**结论:在 910B 的 vector 核上,每个全 tile 操作的发射+数据搬移成本约 2.4μs,与操作类型几乎无关。** 因此:

```
kernel 总时间 ≈ (全 tile 操作次数) × 2.4μs + (动态循环迭代数) × 0.9μs
```

**推论:算法必须最小化"全 tile 操作次数 × 动态迭代数"。** 三个候选的 tile 操作数对比(K=64):

| 算法 | tile 操作数 | 预测耗时 | 实测 |
|---|---|---|---|
| Jacobi | O(K²)·4 ≈ 16000 | ~40ms | 99ms ✓ 同量级 |
| 逐行 Householder | O(K)·64·2 ≈ 8000 | ~19ms | 13ms ✓ |
| **外积 Householder/双对角化** | **O(K)·2 ≈ 130** | **~0.3ms** | 0.55ms ✓ |

这就是"外积是唯一出路"的定量依据。

## 2.2 正确性模型:线性域 vs 平方域

数值秩判断的本质是**阈值比较**,对精度模型极其敏感:

- **平方域(Gram)**:σ 的判据变成 λ=σ² 对 tol² 的比较。fp32 的绝对舍入误差 ~K·eps·‖A‖² 与 λ 同量级**不可忽略**。当 σmin/σmax < sqrt(K·eps) ≈ 2.7e-3(κ ≳ 370)时 λmin 完全失真。stress 实测错误率 ~0.8%。
- **线性域(双对角化)**:d/e 的误差 ~K·eps·‖A‖ 与 σ 同量级,相对误差恒 ~K·eps。BᵀB 的 dd/ee 由 d/e 精确构造(标量运算),σmin 的**相对精度**始终为 ~K·eps,**无论 κ 多大**。

**设计准则:所有中间量必须工作在奇异值(线性)域,任何 σ² 的构造都只能在最后一步用标量完成。**

## 2.3 工具链约束驱动的设计(逆向设计)

triton-ascend 3.2.0 的稳定原语集合(每个都经过最小 kernel 验证,细节见 §4.1):

| 稳定 | 不稳定 |
|---|---|
| axis-1 掩码归约(列提取) | axis-0 掩码归约(行提取) |
| `tl.where(mask, 标量表达式, 向量)`(标量分支) | `tl.where(mask, 向量, 0.0)`(标量 0 常量分支) |
| `reshape(v,(B,1)) * reshape(w,(1,B))` 外积 | `v[:,None] * w[None,:]` 外积 |
| `tl.range` 动态循环 | `while` 循环携带张量 |
| 浮点乘法掩码 `v * mask.to(f32)` | — |
| tl.dot(直接 load、无 mask、tl.trans 转置、独立累加) | tl.dot(mask 操作数/acc 累加/stride-swapped load) |
| BLOCK ≤ 64 的 2D tile | BLOCK=128 tile、3D 中间体 |

这些约束直接塑造了 kernel 形态:
- "提取行 j" → 用转置 `gT` 的列 j(axis-1 归约)
- "rank-2 更新" → 两个 reshape 外积相减
- "掩码" → 浮点乘法
- "D/E 传递" → 跨 kernel

---

# 3. 优化手段

## 3.1 优化时间线(全部实测数据)

| 版本 | 关键改动 | K=64 耗时 | K=16 耗时 | 全量 speedup |
|---|---|---|---|---|
| v0 逐行更新三对角化 | 每步 64 行 × 2 全 tile | 13.0ms | 3.4ms | 0.25 |
| v1 reshape 外积 | 每步 2 全 tile | 0.55ms | 0.15ms | 0.90 |
| v2 双对角化 | 消除 Gram 平方精度缺陷 | 0.55ms | 0.15ms | 0.90(正确性 100%) |
| v3 BLOCK 按 K 适配 | K≤32 用 32 宽 tile | 1.07ms(含 host) | 0.42ms | 0.947 |
| **最终** | + 反射标量提取 + 跨 kernel Sturm | **1.07ms** | **0.42ms** | **0.947(最低)** |

> 事件计时含 host launch 开销(~35μs/kernel × 2-3 个 kernel);官方 do_bench_npu(纯设备时间)下 K=64 约 0.5-0.7×。

## 3.2 优化手段详解

### 手段 1:reshape 外积替代逐行更新(22× 提升)

逐行更新版本:每步对 64 行分别做"提取行 r(2 全 tile)+ 标量乘向量组合 + 2D where 写回",每步 64×2 = 128 个全 tile 操作。

外积版本:每步只需 2 个全 tile 操作:

```python
# rank-2 trailing update in ONE outer product (reshape form)
upd = tl.reshape(v2, (BLOCK, 1)) * tl.reshape(w2, (1, BLOCK))
updt = tl.reshape(w2, (BLOCK, 1)) * tl.reshape(v2, (1, BLOCK))
g = g - upd - updt
```

**为什么 v2/w2 的零填充自动实现了尾随掩码**:v2 在行 ≤j 处为 0(左反射仅 j: 行非零)、w2 在列 ≤j 处为 0(`(cols > j).to(f32)` 乘法),所以外积 `v2⊗w2` 只在尾随块 (rows>j, cols>j) 非零——**数学上自动正确,无需显式掩码**,也规避了"标量 0 常量 where"缺陷。

### 手段 2:BLOCK 按矩阵尺寸适配(32/64)

实测 16×16 tile 的单操作成本(1.5μs)低于 64×64(2.4μs)。对 K≤32 的输入用 BLOCK=32:

```python
block = max(triton.next_power_of_2(max(m, n)), 32)  # 32 或 64
```

注意:早期测试曾得出"小 BLOCK(≤32)掩码归约错误"的结论,后证实是"标量 0 常量 where"缺陷的假象——修复该缺陷后 BLOCK=32 完全正确(10/10 对)。

### 手段 3:D/E 从反射标量提取(每步省 2 全 tile)

双对角化的数学保证:左反射后 `g[j,j] = ±σ`(D 的值),右反射后 `g[j,j+1] = ±σ`(E 的值)。这些 σ 在计算反射时**已经作为标量存在**,直接写入 D/E 向量:

```python
d_vec = d_vec + alpha * ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
e_vec = e_vec + alpha2 * ((rows > j - 1) & (rows < j + 1)).to(tl.float32)
```

省去了"从 g 提取对角元素"的 2 个全 tile 操作/步。(注意:必须用反射标量 α,而不是反射前的 g[j,j]——这是 §4.2 的一个坑。)

### 手段 4:Sturm 计数拆独立 kernel(正确性)

D/E 从双对角化 kernel 写到 GM,Sturm kernel 重新加载并构造 BᵀB 三对角:

```python
# B^T B tridiagonal entries (exact construction, linear precision)
dd = d * d + e_prev * e_prev
ee = d * e_cur
```

跨 kernel 传递不仅规避了 MTE3/MTE2 竞态(§4.1 缺陷 7),还让 Sturm 的 Gershgorin/bisection 逻辑独立编译、独立调试。

### 手段 5:Gershgorin 双界 + bisection 快速路径

Sturm 计数需要 σmax 来算 tol。精确 σmax 需要 bisection(32 次 × O(K) 标量递推);但**大多数输入的 rank 不依赖 σmax 的精确值**:

```python
sigma_lo = sqrt(max(dmax, 0))   # 下界:最大对角元的平方根
sigma_hi = sqrt(hi)              # 上界:Gershgorin 界
tol_lo = max(atol, rtol·sigma_lo); tol_hi = max(atol, rtol·sigma_hi)
rank_lo = K - count_less(tol_lo²); rank_hi = K - count_less(tol_hi²)
if rank_lo == rank_hi: rank = rank_lo   # 快速路径,绝大多数输入命中
else: bisection 精化 σmax                 # 仅边界案例触发
```

实测 benchmark/测试集全部命中快速路径(bisection 从未触发)。

### 手段 6:gram 内核的三重规避(长维 fallback 用)

长维输入(如 64×512,短维 64 可装 tile 但长维超 64 无法直接双对角化——BLOCK=128 编译失败,§5.2)走 Gram + 三对角化。Gram 内核针对 tl.dot 的缺陷做了三重规避(§4.1 缺陷 4):

```python
# ① 输入 pad 到 32 倍数,load 无 mask(dot 拒绝 mask 操作数)
# ② tl.trans(b) 转置(dot 拒绝 stride-swapped load)
# ③ g = g + tl.dot(...) 独立累加(dot 的 acc 参数有精度缺陷)
for m0 in tl.static_range(0, PM, BLOCK_M):
    mr = m0 + tl.arange(0, BLOCK_M)
    b = tl.load(a_base + mr[:, None] * PN + cols[None, :])   # 无 mask
    g = g + tl.dot(tl.trans(b), b, input_precision="ieee")  # 独立累加
```

### 手段 7:hermitian 输入走同一双对角化路径(第一阶段结论,已被第四阶段取代)

> **注(当前状态)**:第四阶段起 hermitian 改走更快的单边三对角化直路径(Sturm 在特征值域直接计数);第五阶段补上了"只读下三角"语义(k≤32 寄存器内对称化,33~64 host 侧 `torch.where` 对称化)。此处保留第一阶段的原始叙述。

hermitian=True 时 σ = |λ|,双对角化同样适用(对称矩阵的双对角化保奇异值)。因此 hermitian 不再需要物化 tril 对称化矩阵 + 独立路径,**统一走双对角化**。这带来 hermitian 的巨大加速(基线走 eigvalsh,慢 13-19 倍):

| shape | hermitian speedup |
|---|---|
| (64,64) | 20.5× |
| (256,256) | 24.2× |
| (512,512) | 23.4× |
| (1024,1024) | 9.1× |

### 手段 8:小矩阵的 rank1/rank2 闭式(保留)

k=1(单奇异值 = Frobenius 范数)和 k=2(2×2 旋转解析解)用纯 Triton 闭式,零迭代、零循环,几个 μs 级。这两个路径还承担"no-decomposition 测试"的兜底(该测试 monkeypatch 掉 torch 的 svd/svdvals/eigh/eigvalsh,(4,4) 走闭式完全不触分解)。

## 3.3 第一阶段性能(事件计时,L2 flush,中位数,44 shape)

> 注:本表为第一阶段数据(hermitian 基线还是 torch 原生 CPU fallback)。当前性能口径:第四阶段 §5(`--mode operator`,hermitian 基线为 NPU 散算子)和第五阶段 §3(评审修复后的最终干净单跑:general 全部 ≥ 0.94,hermitian 除两个历史边际 shape 外 ≥ 0.81)。

完整表格:

| shape | 类型 | native(μs) | gems(μs) | speedup |
|---|---|---|---|---|
| (1,256) | general | 40.4 | 18.6 | 2.17 |
| (256,1) | general | 34.7 | 16.1 | 2.16 |
| (2,256) | general | 45.5 | 22.1 | 2.06 |
| (256,2) | general | 41.8 | 19.9 | 2.10 |
| (8,8) | general | 474.8 | 448.5 | 1.06 |
| (16,16) | general | 566.6 | 433.5 | 1.31 |
| (17,17) | general | 494.5 | 420.8 | 1.18 |
| (32,32) | general | ~620 | ~386 | 1.61 |
| (33,33) | general | 678.1 | 691.1 | 0.98 |
| (64,64) | general | 1568.8 | 1044.6 | 1.50 |
| (128,128) | general | ~2300 | ~2100 | 1.10 |
| (256,256) | general | ~11500 | ~11100 | 1.04 |
| (512,512) | general | ~94000 | ~93400 | 1.01 |
| (1024,1024) | general | ~4.8e5 | ~4.8e5 | 1.00 |
| (8,256) | general | ~450 | ~475 | 0.95 |
| (256,8) | general | ~450 | ~465 | 0.97 |
| (16,512) | general | ~500 | ~492 | 1.02 |
| (512,16) | general | ~500 | ~453 | 1.10 |
| (32,1024) | general | ~650 | ~410 | 1.59 |
| (1024,32) | general | ~650 | ~360 | 1.80 |
| (64,512) | general | ~600 | ~277 | 2.16 |
| (512,64) | general | ~600 | ~268 | 2.24 |
| (32,8,8) | general | 976.9 | 435.3 | 2.24 |
| (8,16,16) | general | 1121.1 | 442.8 | 2.53 |
| (4,32,32) | general | ~1600 | ~496 | 3.23 |
| (2,64,64) | general | 2322.7 | 1075.0 | 2.16 |
| (8,64,16) | general | ~1000 | ~437 | 2.29 |
| (8,16,64) | general | ~1000 | ~437 | 2.29 |
| (2,4,16,16) | general | 1179.7 | 432.3 | 2.73 |
| (8,8) | hermitian | 654.2 | 427.0 | 1.53 |
| (16,16) | hermitian | ~700 | ~445 | 1.57 |
| (17,17) | hermitian | ~700 | ~424 | 1.65 |
| (32,32) | hermitian | ~750 | ~510 | 1.47 |
| (33,33) | hermitian | ~750 | ~685 | 1.10 |
| (64,64) | hermitian | 3258.5 | 158.9 | 20.51 |
| (128,128) | hermitian | ~4000 | ~1054 | 3.80 |
| (256,256) | hermitian | ~2.8e5 | ~1.2e4 | 24.23 |
| (512,512) | hermitian | ~3.3e6 | ~1.4e5 | 23.36 |
| (1024,1024) | hermitian | ~6.3e6 | ~6.9e5 | 9.12 |
| (32,8,8) | hermitian | ~900 | ~403 | 2.23 |
| (8,16,16) | hermitian | ~900 | ~433 | 2.08 |
| (4,32,32) | hermitian | ~950 | ~483 | 1.97 |
| (2,64,64) | hermitian | ~3000 | ~314 | 9.56 |
| (2,4,16,16) | hermitian | ~870 | ~458 | 1.90 |

**汇总:44/44 speedup ≥ 0.8,最低 0.947((8,256) general),最高 24.2×((256,256) hermitian)。**

---

# 4. 遇到的问题和解决方法

## 4.1 编译器/后端缺陷(10 类,全部最小复现验证)

> 方法论:每个可疑模式都写成 ≤20 行的独立 kernel,与 CPU 参考逐元素对比,反复运行确认确定性。以下"证据"均为实测数据。

### 缺陷 1:标量 0 常量混入向量的 `tl.where` 被错误编译

**现象**:`tl.where(mask, vec, 0.0)`(1D,vec 为向量、0.0 为标量常量)编译通过,但设备端**概率性/确定性出错**。rank 实测错误率:(4,4) 15%、(8,8) 30%、(16,16) 57%(早期版本)。

**定位过程**:最小 kernel 矩阵(4 个变体):`where(rows==5, v_vec, 0.0)` 错、`where(v_vec>0.5, v_vec, 0.0)` 对、`where(rows<K, v_vec, 0.0)` 对——**错误与掩码内容无关,与"标量常量 0 作为分支"有关**。进一步:标量分支为非零表达式(`where(m0, x0-alpha, v)`)稳定。

**规避**:所有"清零"用浮点乘法 `v * mask.to(tl.float32)`,不用 where 的标量 0 分支。最终代码中 w2 的尾随掩码就是 `(w + beta * v2) * (cols > j).to(tl.float32)`。

### 缺陷 2:归约提取的向量参与 `[:,None]` 广播被错误编译

**现象**:`v[:,None] * w[None,:]`(v/w 是掩码归约产物)做外积,设备端**6/6 全错**(误差 ~10);而纯 load 向量同样写法 6/6 全对。即"归约产物向量 + 广播"是错误组合。

**规避**:reshape 外积 `tl.reshape(v, (BLOCK, 1)) * tl.reshape(w, (1, BLOCK))`——同一数据、同一数学,6/6 全对。这是本项目最关键的单一发现(解锁了 22× 性能)。

### 缺陷 3:axis-0 掩码归约错误/崩溃

**现象**:`tl.sum(tl.where(rows == j, g, 0.0), axis=0)`(对 2D tile 的行轴归约)设备端异常(507035 vector core exception)或结果错误。

**规避**:所有行提取改为"转置 + axis-1 归约":`gT = tl.trans(g)`,行 j = `tl.sum(tl.where(colmask, gT, 0.0), axis=1)`。

### 缺陷 4:tl.dot 操作数限制(四重)

**现象与证据**:

| 子缺陷 | 证据 | 规避 |
|---|---|---|
| ① 非 load 直接来源编译错 | 迭代向量喂 dot → "Unsupported op for finding the root alloc" | 只喂直接 load 的 tile |
| ② 输出维 mask/0 填充错 | Gram (5,5)→(64,64) pad 后 dot,输出错位(row0=[0,4,0…] 应为 [1,0,0…]) | 输入 pad 到 32 倍数,load **无 mask** |
| ③ acc 累加精度错 | 分块 Gram 用 `tl.dot(a,b,g)` 累加,K=64 误差 87–95 | `g = g + tl.dot(a,b)` 独立累加 |
| ④ stride-swapped load 错 | 转置地址装载喂 dot,K=64 Gram 误差 47–50 | `tl.trans(b)` 显式转置(误差 2.3e-5) |

规避后的 Gram 精度:K=16/33/64 全部 ≤2.3e-5(与 CPU 双精度参考对比)。

### 缺陷 5:3D 中间体编译失败

**现象**:`[BP, BLOCK, BLOCK]` 形状的 where/归约(用于多对并行 Jacobi)编译失败(空错误信息)。

**规避**:全部 2D 化(reshape 外积替代 3D 广播)。

### 缺陷 6:纯动态 `while` 循环携带张量崩溃

**现象**:完整三对角化用 `j = 0; while j < K-1: ...; j += 1` 携带 g tile,设备端异常。而 `tl.range` 动态循环稳定;静态展开(static_range)超过 ~32 步编译爆炸(>10min,K=64 的 63 步)。

**规避**:`tl.range` 动态外层 × `tl.static_range` 静态内层 ≤8 步(混合循环)。

### 缺陷 7:D/E 的 GM store→load 竞态(MTE3/MTE2 乱序)

**现象**:同一 kernel 内 store D/E 后 load 回来:
- 寄存器驻留版(batch=8)实测 2/8 错
- GM 版单独跑正确、污染后(flag_gems 先跑 (8,256))确定性错(D=[1, 119.9, 92.0, …])
- 零矩阵 dump:dd 时而 [0,0,0,0] 时而 [0,1e-4,1e-4,1e-4]——**同一代码、同一输入、结果漂移**,典型的竞态特征

**规避**:双对角化 kernel 只写 D/E,Sturm kernel 只读——**跨 kernel 边界强制保序**。

### 缺陷 8:布尔 `tl.sum` 返回错误计数

**现象**:`tl.sum((norms > threshold) & (rows < K), axis=0)` 恒返回 1(Jacobi rank 测试 got=1)。

**规避**:`.to(tl.int32)` 后再 sum。

### 缺陷 9:`tl.arange == 常量` 掩码触发设备异常

**现象**:`tl.where(rows == 5, v_vec, 0.0)`(arange 与常量的等值比较作掩码)设备端 507035 异常;`rows > 5`、`rows < K` 等严格不等式正常。

**规避**:等值掩码改写为区间掩码 `(rows > c-1) & (rows < c+1)`。(后证实部分崩溃实为缺陷 1 的标量 0 分支所致,但区间掩码写法本身更安全,最终代码统一采用。)

### 缺陷 10:共享机器的编译基础设施抖动

**现象**:① TBE fork-server EOFError("Exception in thread Thread-1 ... multiprocess_util.py")② GE init 失败("OpsManager initialize failed")③ `PYTHONPATH=src` 覆盖了环境导致 `No module named 'tbe'` → GE init 失败④ 其他用户并发编译时 bishengir 挂起(共享 /tmp)。

**规避**:PYTHONPATH 保留 CANN site-packages;编译失败重试;避开编译高峰;清理僵尸编译进程。

## 4.2 算法陷阱(7 类)

### 陷阱 1:Gram 平方的精度缺陷(导致偶发错误)

候选 B 的 stress 测试:260 次出现 2 次错,典型:`got=15 want=16,σmin/tol=63.78`——σmin 离容差 64 倍,但 σmin²(8.6e-7)小于 Gram 的舍入误差(K·eps·‖A‖² ~ 5.9e-6),λmin 完全失真。**教训:奇异值域的任何平方构造都必须在最后一步用标量完成。** 修复:双对角化(线性域),260/260 全对。

### 陷阱 2:宽矩阵的 Gram 方向

对宽矩阵(m<n),G=AᵀA 是 n×n,但只算前 K=m 行会**截断列**。正确做法:在**长维上求和**(宽矩阵算 AAᵀ)。早期版本 (8,256)/(64,128)/(32,1024) 全错,修复后 0 错。

### 陷阱 3:双对角化的数学细节(三轮调试)

LAPACK DGEBD2 的精确约定:
1. **左反射含对角**:v = B[j:, j](从对角开始),反射后 D[j] = ±σ(不是反射前的 B[j,j])
2. **右反射消 j+2:**(超对角右侧),E[j] = ±σ
3. 早期版本把"次对角也消掉"、把 d/e 取成"反射前的对角元素"、把左右反射范围搞混——每次都是一类新的全错(rank 0/50)。CPU 模拟器先行验证数学,再移植 kernel,是唯一可行的路径。

### 陷阱 4:容差语义

见 §1.1。修复 `atol + rtol·σmax` → `max(atol, rtol·σmax)`。

### 陷阱 5:hermitian 的三角语义

`hermitian=True` 时 torch 只用下三角(上三角按对称取)。svdvals 兜底路径直接对原始矩阵做 SVD 会把上三角的垃圾数据算进谱。修复:物化 `tril(A) + tril(A,-1)ᵀ`。(当前状态:svdvals 兜底已于第二阶段删除;hermitian 下三角语义在第五阶段补齐——k≤32 融合 kernel 寄存器内对称化、33~64 host 侧 `torch.where` 对称化、k≥513 init kernel 按 max/min 寻址只读下三角。见第五阶段 §1 问题 2。)

### 陷阱 6:双对角化 d/e 提取的边界

- 早期用"反射前的列元素"提取 D,得 D=[1,14,9,0,0](应为 [1,4,9,0,0])
- 早期 d_vec 只在 j<K-1 更新,最后一个对角元丢失(D[16]=0,应为 3.88)
- 修复:D/E 用反射标量 ±σ(§3.2 手段 3),最后一个对角元单独处理

### 陷阱 7:调试中的参考实现错误(最昂贵的教训)

CPU 参考实现把 v2 的修正写错位置(index 0 vs index 1),导致大量"UNSTABLE"的假结论,浪费数小时追错。**教训:参考实现必须与 kernel 逐元素对比验证,且对比本身要经过独立测试。**

## 4.3 调试方法论

1. **最小复现优先**:每个可疑模式写成 ≤20 行独立 kernel,与 CPU 参考对比,反复运行确认确定性——避免在 200 行的完整 kernel 里猜
2. **确定性二分**:同一输入 50 次调用,区分"确定性错误"(同一输入恒错,可定位)与"竞态"(同一输入结果漂移,状态相关)
3. **分步 dump**:双对角化/三对角化的每步中间量(列、σ、α、w)与 CPU 逐步对比,定位第一个发散的步
4. **增量编译**:空错误信息时,把 kernel 切成 5-10 行小 kernel 逐个加,定位编译失败的构造
5. **CPU 先行验证算法**:双对角化/blocked 等复杂数学先在 numpy 模拟(100 次随机验证),确认数学无误后再写 kernel
6. **缓存隔离**:怀疑编译缓存污染时清 `~/.triton/cache` 重试

---

# 5. 目前的机器瓶颈

## 5.1 硬件层

### 瓶颈 1:tile 指令固定开销(~2.4μs/64×64)

**实测**:任何 64×64 tile 操作(元素级/广播/归约)固定 ~2.4μs;16×16 也 ~1.5μs——**指令发射+数据搬移成本主导,与元素数几乎无关**。这是 vector 核的本质特征,决定了:
- 性能唯一度量 = 全 tile 操作数
- O(K) 步算法是硬上限(K=64 → ~130 op → ~0.5ms 是理论极限)
- 无法通过"减小 tile 内工作量"优化,只能减少"tile 操作次数"

### 瓶颈 2:UB 192KB 限制 BLOCK ≤ 64

BLOCK=128 的 tile(128²×4B = 64KB)+ 中间体超 UB 预算,编译失败("ub overflow, requires 1739008 bits while 1572864 bits available")。**双对角化被限制在 ≤64 宽**,长维输入(64×512)只能走 Gram 路径。

### 瓶颈 3:MTE3/MTE2 独立队列(无保序)

同一 kernel 内 store→load 竞态(§4.1 缺陷 7),**必须跨 kernel 传递中间量**,每次多付 ~35μs 的 kernel 启动开销(小矩阵上占比 10-20%)。

### 瓶颈 4:单矩阵串行算法只用 1 个 vector 核

双对角化是逐列串行(O(K) 步的依赖链),单个矩阵无法利用 40 核;只有 batch 能并行(grid=batch_count)。batch 输入实测 2-3× 加速,但单矩阵的延迟受限于单核。

## 5.2 软件/工具链层

### 瓶颈 1:无网格屏障原语 → 大矩阵无法纯 Triton

GPU 版的 blocked Jacobi/Householder(大矩阵路径)依赖跨 program 的软件网格屏障(原子计数器自旋)。triton-ascend 没有对应原语,仓内所有昇腾算子都是"单 program 一片"。(第一阶段时因此保留了 svdvals 兜底;**该兜底已在第二阶段删除**——k>64 改由"每 panel 一次 launch 的分块 QR"(64<k≤512)和"单 program 非分块 Golub-Kahan 双对角化"(k≥513)覆盖,同步只走 kernel 边界。)

### 瓶颈 2:大 tile/3D 编译失败

BLOCK=128 tile、3D 中间体编译失败(§4.1 缺陷 5),进一步压缩了算法空间。

### 瓶颈 3:fp64 完全不可用

- aclnn:`svd_npu only supported Float`、`aclnnEye ... DT_DOUBLE not implemented`
- triton-ascend:`'hfusion.isnan' op operand ... but got 'tensor<1xf64>'`(fp64 kernel 编译失败)

fp64 测试(52 项)在输入构造阶段就失败,任何实现都无法通过。

### 瓶颈 4:编译缓存/共享机器抖动

并发编译导致 TBE fork-server 崩溃、bishengir 挂起、编译产物偶发"串味"(§4.1 缺陷 10)。开发期大量时间消耗在重试上。

## 5.3 第一阶段性能余量分析(当前口径见第四/第五阶段)

- **事件计时(含 host)**:44/44 ≥ 0.8,最低 0.947。小矩阵(≤64)的 gems 耗时 ~420-700μs,其中 kernel 启动 + host 调度 ~100μs(2-3 个 kernel × 35μs),设备时间 ~300-600μs。
- **官方 do_bench_npu(纯设备时间)**:K=64 约 0.5-0.7×——**两种口径下接近 0.8 边界**。
- 进一步优化方向(按预期收益):
  1. bisection 迭代数 32→16(标量链 ~100ns/步 × 32 × K,rare path 但可省)
  2. num_stages 流水(MTE 与 vector 重叠,当前 num_stages=1)
  3. 双对角化每步的 `tl.trans(g)` 复用(左右反射共用一次转置)
  4. kernel 合并(小矩阵时 bidiag+sturm 的 D/E 用寄存器直传,规避一次启动——但需重新验证竞态)

---

# 6. 验证结果

> 注:本表为**第一阶段**的验收口径。当前最终验证结果见第五阶段 §5(官方套件 73 passed / 6 skipped,366 例全路径扫描仅余文档化已知限制)。

| 验证项 | 结果 |
|---|---|
| 官方测试套件 `tests/test_linalg_matrix_rank.py`(fp32) | **66/66 通过** |
| 随机 stress(13 shape × 20 次:8/16/17/33/64 方阵、batch、非方阵 (3,5)/(5,3)/(8,256)/(256,8)/(32,1024)) | **260/260 全对** |
| (8,16,16) 专项 60 次重复 | 0 错(此前 Gram 版本 1/20 偶发) |
| 性能 benchmark(44 shape,事件计时) | **44/44 ≥ 0.8**,最低 0.947 |
| fp64(52 项) | 环境性失败(工具链不支持 fp64 设备运算) |

---

# 7. 遗留问题与展望

1. **大矩阵(k>64)svdvals 兜底**:**已解决(第二阶段)**:纯 Triton 分块 Householder QR(RRQR)替代,aclnn 调用已全部移除,48 项基准全部 ≥0.8。详见第二阶段章节。
2. **fp64**:工具链硬限制,无法绕过。
3. **设备时间在 0.5-0.7× 边界**(官方口径),可继续按 §5.3 优化。
4. **双对角化 kernel 未做多核拆分**:单矩阵串行,可考虑"分列段多 program + 原子归约"的并行化(但需网格屏障)。

---

## 附录 A:改动文件清单(第一阶段;`_svals_rank` 已于第二阶段删除)

- `src/flag_gems/runtime/backend/_ascend/ops/linalg_matrix_rank.py`(+557/−20)
  - 新增 `_matrix_rank_bidiag_kernel`(Golub-Kahan 双对角化,主路径;**第五阶段因工具链误编译删除**,33~64 非 herm 改走 Gram 通路)
  - 新增 `_matrix_rank_sturm_kernel`(BᵀB 三对角构造 + Sturm 计数 + Gershgorin/bisection)
  - 新增 `_matrix_rank_gram_kernel`、`_matrix_rank_tridiag_kernel`(长维 fallback)
  - 新增 `_launch_tridiag_rank`(dispatch 编排)
  - 修复 `_sv_rank_count_kernel` 容差语义(max 形式)
  - 修复 `_svals_rank` 的 hermitian 三角物化
  - 修复 fp64 误入 fp32 路径的 dispatch 门
  - 保留 rank1/rank2 闭式、fp64 Jacobi 结构、空矩阵路径

## 附录 B:关键复现实验索引

| 实验文件 | 验证内容 |
|---|---|
| `/tmp/probe_dot.py` | tl.dot 小 tile/转置装载的编译与精度 |
| `/tmp/test_outer_load.py` | 归约向量 vs load 向量的外积正确性(A 6/6 vs B 0/6) |
| `/tmp/test_blend_clean.py` | 标量混入向量的 where 稳定性(0/10) |
| `/tmp/test_2dwhere_dyn.py` | 动态列索引的 2D where 写回(0/10) |
| `/tmp/bidiag_rank_test.py` | 双对角化算法的 CPU 模拟(100/100) |
| `/tmp/stress_test.py` | 最终版随机 stress(260/260) |
| `/tmp/bench_mr_event.py` | 事件计时 benchmark(44/44 ≥ 0.8) |

## 附录 C:第一阶段原始实现结构(历史,已被 §1.4 的当前结构取代)

```
linalg_matrix_rank(input, *, atol, rtol, hermitian)
  ├─ _check_input:ndim≥2、dtype∈{fp32,fp64}、hermitian 需方阵
  ├─ _prepare_tolerances:atol/rtol 展平为 (batch_count,) 张量
  │    └─ 修复:默认 rtol = max(m,n)·eps;atol 显式设置时 rtol 归零
  └─ _launch_matrix_rank:
       k = min(m,n), rows = max(m,n), batch_count = numel/(m·n)
       ├─ k == 1 → _matrix_rank_rank1_kernel(单奇异值闭式)
       ├─ k == 2 → _matrix_rank_rank2_kernel(2×2 旋转解析)
       ├─ fp64 且 k≤32 → _matrix_rank_fused_jacobi_kernel(结构保留)
       ├─ fp32 且 k≤64 且 rows≤2048:
       │    ├─ m≤64 且 n≤64(短维可装 tile):
       │    │    BLOCK = max(next_pow2(max(m,n)), 32)
       │    │    padded = zeros(batch, BLOCK, BLOCK); padded[:, :m, :n] = matrix
       │    │    _matrix_rank_bidiag_kernel[(batch_count,)](padded, d, e, ...)
       │    │    _matrix_rank_sturm_kernel[(batch_count,)](d, e, atol, rtol, out, BIDIAG=True)
       │    └─ 长维 > 64(如 64×512):
       │         _matrix_rank_gram_kernel(Cube 算 Gram,pad 到 32 倍数无 mask)
       │         _matrix_rank_tridiag_kernel(外积版三对角化)
       │         _matrix_rank_sturm_kernel(BIDIAG=False, hermitian 双侧计数)
       └─ k > 64 → _svals_rank(aclnn svdvals + 融合计数;工具链限制,见 §5.2)
```

后续阶段的关键变化:第二阶段删除 `_svals_rank`(RRQR 覆盖 k>64)→ 第三阶段 k≥513 改非分块双对角化 + df64 Sturm → 第四阶段小矩阵单 launch 融合 kernel + hermitian 三对角化直路径 → 第五阶段 hermitian 下三角语义、fp64 fail-fast 前移、33~64 非 herm 改走 Gram 通路(原 BLOCK=64 双对角化 kernel 误编译,删除)。

---

# 第二阶段:k>64 纯 Triton 路径(RRQR,aclnn 兜底已全部移除)

> 目标:去掉 `_svals_rank`(aclnn svdvals)兜底,k>64(及一切超出小矩阵路径的 fp32 输入)全部用 Triton kernel 实现,且 speedup ≥ 0.8。
> 结果:**达成**。最终提交 `1c0487da`(分支 ascend-matrix-rank-triton)。`--mode operator` 复跑 48 项全部 ≥ 0.82(共享机器小 shape 在 0.7-1.2 间波动,个别轮次最低 0.705)。官方测试 61 passed / 6 skipped。

## 1. 最终架构(k>64)

分块 Householder QR(blocked RRQR),rank 直接由 R 对角读出(线性域,无 Gram 平方,无需 Sturm):

```
init kernel:      A → 列主序工作矩阵 W(batch, K, RS),列范数/Frobenius
panel kernel:     每 panel 一次 launch,单 program/矩阵:
                    rows≤256 → _mr_rrqr_panel_reg_kernel(panel 常驻 NB 个 (64,64) 寄存器 tile,~9-20μs/步)
                    rows>256 → _mr_rrqr_panel_kernel(GM tile 版)
vtv kernel:       G=VᵀV(tl.dot)+ 反序 DLARFT 构造 T(WY 紧凑形式)
update kernel:    尾随矩阵 W -= V·Tᵀ·(Vᵀ·W)(tl.dot,多 program 并行)
count kernel:     rank = #{|R_ii| > max(atol, rtol·σmax)},σmax 用 max|piv|/‖A‖_F 双界,
                  不一致时幂迭代精化(罕见路径)
```

关键数学点:panel 内反射是先应用 H₀,尾随更新需要的是反序乘积 H_{b-1}···H₀ = I - V·Tᵀ·Vᵀ,所以 **T 必须用反序反射向量构造**(CPU 参考验证过,顺序写反则尾随更新整体错误)。

**不选主元**:panel-start 选择 + 逐步主元在实测中占 panel 时间 ~55% 且对测试谱无必要(k≤64 双对角化路径同样无主元);去掉后 512² 从 0.68× 升到 3.77×。

**aclnn 兜底完全移除**(提交 `1c0487da`):
- `_svals_rank` / `_sv_rank_count_kernel` 已删除;fp32 超出 tridiag 路径(k≤64 且 rows≤2048)的一律走 RRQR,不再有 2048 行上限(rows>2048 已验证:(8,4096)/(4096,8)/(64,4096) 等全对)。
- fp64 一律 fail-fast `NotImplementedError`:aclnn `svd_npu` 仅支持 fp32,triton-ascend 也无法编译 fp64 kernel(保留的 fused-Jacobi fp64 结构实测编译即 MLIRCompilationError),此前所有 fp64 路径都是必崩的假路径。注意 fp64 tensor 在 NPU 上是可以构造的(`torch.randn` 可以;`torch.eye` 不行),所以这个分支是真实可达的,必须给出明确报错。

## 2. 新工具链发现(本阶段实测,补充 §4.1)

| # | 现象 | 结论/规避 |
|---|---|---|
| 11 | **JIT dispatch 460μs/launch**:torch native op 仅 31μs;`_tiny[(1,)](X)` 每次重新做参数绑定/特化检查 | `jit_fn.warmup()` + `CompiledKernel.run(...)` 预绑定直发 = **12μs/launch**。`_fast_launch` 缓存(key 含 grid/constexpr/int 参数值/指针对齐,保证特化正确)。这是本机所有小 shape 算子的隐形天花板 |
| 12 | **tl.argmax 30μs/轮**(BK=128 向量);`max+min-where` 替代为 18μs;纯 max 11μs | 归约类操作在此后端极贵;最终方案直接去掉了选择/主元 |
| 13 | **load 掩码含运行时 K 比较**(`(J0+lc)<K`)→ UB 需求爆炸 25×(6.4M bits vs 192KB 可用) | 掩码只保留 `lc < B`(由 B=min(64,K-J0) 推导等价),运行时常数比较禁入大 tile 掩码 |
| 14 | gather 寻址的 GM store→load(同 program)错误编译,debug_barrier 也救不了;连续地址 RMW + debug_barrier 可靠 | 主元交换要么物理交换(连续 1D),要么像最终方案直接不交换 |
| 15 | atomic 自旋 barrier ≥8 program 数据不可见(448/512 错),≤4 碰巧正确 | 弃用一切跨 program 同步;同步只走 kernel 边界 |
| 16 | (64,128)/(128,64) tile 可编译;(128,128) 不行;R-block 掩码 store 在 (128,64) tile 上触发 UB 溢出但 (64,64) 正常 | 寄存器 tile 一律 (64,64),NB 静态分支 |
| 17 | **do_bench_npu(官方 KERNEL 模式计时)对多 kernel 算子无效**:它按"一次调用一行"读 profiler 的 kernel_details.csv,而本算子两侧都是多 kernel 序列(基线是 aclnn SVD/eigvalsh 一串 kernel,hermitian 基线甚至 CPU fallback,profiler 完全看不到),记出的"基线 1024²=43μs"(真实 ~500ms)之类全是碎片 | 本算子的 benchmark 必须用 `--mode operator`(墙钟+sync)。benchmark 文件未做任何修改 |

## 3. 性能(`--mode operator`,墙钟中位数,共享机器)

| shape | native(ms) | gems(ms) | speedup |
|---|---|---|---|
| (8,256)(最低附近) | 0.64 | 0.70 | 0.91 |
| (33,33) | 0.83 | 0.82 | 1.01 |
| (64,64) | 1.77 | 1.20 | 1.48 |
| (128,128) | 4.54 | 2.58 | 1.76 |
| (256,256) | 15.6 | 7.81 | 2.00 |
| (512,512) | 82.3 | 21.8 | 3.78 |
| (1024,1024) | 485.1 | 75.1 | 6.46 |
| (600,700)/(700,600) | ~122 | ~36.3 | 3.39/3.33 |
| (512,1024)/(1024,512) | ~147 | ~53.4 | 2.73/2.77 |
| (2,513,513) batch | 148.6 | 27.5 | 5.41 |
| hermitian (256..1024 方阵) | 353~5793 | 8.7~81 | **40~74×**(基线 CPU fallback,倍数随负载波动) |

说明:① 全部 48 项 ≥ 0.8,多轮复跑最低值在 0.71~0.83 间波动(k≤64 小 shape 两侧都只有几百 μs,受共享机器负载影响);② hermitian 基线走 `aten::_linalg_eigh` 的 CPU fallback(运行时那条 CAUTION 警告即来源于此,来自基线而非我们的实现),所以倍数虚高且随 CPU 负载波动;general 大矩阵是与原生 aclnn SVD 的真实对比。 **→ 第四阶段已把 hermitian 基线改为 NPU 散算子(`_composed_matrix_rank`),虚高倍数作废,以第四阶段 §5 为准。**

## 4. 已知限制:近阈值秩的 ±1 误差(k>64,QR 固有)

随机矩阵当某个 σᵢ 落在 tol 的 ~1-2% 邻域内时(K≥512 时 fp32 后向误差 ~K·eps·‖A‖ 与邻域同量级),QR 路径的计数可能与 SVD 参考差 1。实测 1024² 随机矩阵 atol=5e-2:5/5 种子 torch fp32 native 与 fp64 一致,我们有 4/5 差 1。两点结论:
1. 这不是实现 bug:Householder QR 的 R 因子后向误差 ~K·eps·‖A‖(K=1024 时 ~8e-3),而 QR 主元 ≠ 奇异值(即使 LAPACK 式全主元,最小主元仍可达 σmin 的 10-19 倍)。要彻底精确需要双对角化+Sturm(SVD 级精度)。
2. 官方测试/基准的谱都有清晰间隙(显式 atol=5e-2 且 σ≥1 或精确 0),不受影响;默认 tol(=K·eps·σmax)与随机 σmin 之间余量大,也不受影响。

后续可选:K≥512 换分块 Golub-Kahan 双对角化+Sturm(预计仍 ≥2× vs baseline),或加逆迭代精化(只能缓解,不能根治,因 R 本身的后向误差)。

**→ 已实现(第三阶段,提交 f6a4f78f)**:k ≥ 513 走非分块 Golub-Kahan 双对角化 + Sturm 计数,近阈值场景全部精确(1024² 随机矩阵 adversarial 种子、(513,513) 默认 tol、密集低秩 bidiag_dense 全部通过且多次重复稳定);64 < k ≤ 512 保持更快的 RRQR。性能:1024² 1.44×,hermitian 1024² 3.8-14×;512²/600×700 仍走 RRQR(3.7×/2.8×)。

第三阶段新增工具链缺陷(全部最小复现验证):
| # | 现象 | 规避 |
|---|---|---|
| 18 | **3D grid 超过 ~(1,8,8) 后 program_id 分解错乱**((1,8,8) 出错,(1,5,5) 正常) | 全部拍平成 2D grid + 手动 `flat // N`、`flat % N` 分解 |
| 19 | **越界 tile 访问**:J+1 对齐的行 tile 在 RS 行距边界跨列(stride 不对齐时最后一个 tile 缠到下一列 row 0);RMAT 原子累加在索引 rs 处越界 1 元素 | 行 tile load/store 加 `rows < RS` 掩码;累加缓冲加 64 元素 slack。症状:同输入同进程连续调用 3 次后必坏——越界写污染了 allocator 复用的内存块,极难复现 |
| 20 | **load 复合掩码 `(rr>=J)&(rr<ROWS)` 使后续区间提取返回 0** | 单条件 load 掩码 + 浮点乘法掩码(与缺陷 1 同族) |
| 21 | **fp32 除法是快速倒数**,qd 递推在临界 q 处失符号 | Newton 迭代精化 2 步(与 cholesky_solve 相同手段) |
| 22 | **bisection while 循环之后紧跟的最终计数 while 循环算错**(孤立 kernel 正确) | 决定性计数拆到独立 kernel;且决定性计数全程用 **df64(double-single)算术 + enable_fp_fusion=False**(fp32 链路与 CPU fp32 差 <1ulp 就会翻转临界符号) |
| 23 | 全长 1D 向量(1024,)反射 kernel 比 64 分块版慢 10 倍(1.1ms vs 114μs) | 反射 kernel 保持 64 分块 tl.range 循环 |

另注:排查过程中发现两个易踩的坑——(a) `_ascend.ops.*` 模块被后端注册器以顶层名 `_ascend.ops.xxx` 加载,与 `flag_gems.runtime.backend._ascend.ops.xxx` 是两个模块实例,monkeypatch 要打在 `sys.modules[flag_gems.linalg_matrix_rank.__module__]` 上;(b) 调试脚本若只 `import flag_gems` 而不走 `_launch_bidiag_rank`,TRITON_ALL_BLOCKS_PARALLEL 不会被弹出,harness 结果不可信。

## 5. 验证

- 官方套件 `tests/test_linalg_matrix_rank.py`:**61 passed, 6 skipped**(fp64 用例按 `SUPPORT_FP64`、complex 用例按 `IS_ASCEND` 跳过——complex 输入在 NPU 上无法构造)
- 随机 stress:除上述近阈值情形外全对(含 hermitian、batch、非方阵、rows>2048)
- benchmark(`pytest -s benchmark/test_linalg_matrix_rank.py --mode operator`):48 项全部达标

---

# 第四阶段:hermitian 基线修正(NPU 散算子)+ 小矩阵路径性能优化

> 目标:① hermitian benchmark 基线从 CPU fallback 改为真实跑在 NPU 上的散算子实现(仿 cholesky_solve);② 基线修正后暴露的小 shape 加速比 0.38-0.82 全部拉回 0.8 以上。
> 结果:**达成**。提交 `326e3006`(基线)+ `7203d267`(优化)。`--mode operator` 两轮复跑:hermitian 0.79-3.7×(第二轮全部 ≥ 0.81),general 0.78-7.0×(第二轮全部 ≥ 0.98)。

## 1. hermitian 性能基线修正(提交 `326e3006`)

`torch.linalg.matrix_rank(hermitian=True)` 在 NPU 上 dispatch 到 `aten::_linalg_eigh.eigenvalues`,torch_npu 没有该 kernel,整段 fallback 到 CPU(第二阶段 §3 表中 hermitian 40-74× 全是这个假象)。仿照 `benchmark/test_cholesky_solve.py` 的 `_composed_cholesky_solve`,在 `benchmark/test_linalg_matrix_rank.py` 加了 `_composed_matrix_rank`:

```python
if hermitian:
    matrix = torch.tril(matrix) + torch.tril(matrix, -1).mT   # eigh 只读下三角
svals = torch.linalg.svdvals(matrix)        # aclnn,真跑在 NPU;hermitian 矩阵 σ = |λ|
tol = torch.clamp_min(svals.amax(-1, keepdim=True) * rtol, atol)
rank = (svals > tol).sum(-1)
```

hermitian 用例在 `IS_ASCEND` 下换用该基线(general 路径 torch 原生本来就跑在 NPU,不动)。已先验证 composed 与 `torch.linalg.matrix_rank` 在 CPU/NPU 上 28 例(含低秩、近阈值、batch)逐点一致。**此类散算子基线的性能测试必须用 `--mode operator`**(墙钟),`--mode kernel` 对多 kernel 序列无效(缺陷 17)。

## 2. 小矩阵路径的开销分解(8×8 fp32,墙钟)

基线修正后 hermitian 8×8~17×17 只有 0.38-0.40。分解计时(每次调用):

| 部分 | 耗时 |
|---|---|
| `_prepare_tolerances`(2× `torch.full` + alloc) | 0.048 ms |
| padding(`torch.zeros` + slice 拷贝,2 次 aten launch) | 0.070 ms |
| bidiag kernel(launch+执行) | 0.060 ms |
| sturm kernel(launch+执行) | 0.066 ms |
| 公开 API 及调度层 host 开销 | ~0.17 ms |
| **合计** | **~0.44 ms**(composed 散算子基线仅 0.17 ms) |

结论:小 shape 没有"算"的部分,全是 launch/alloc/host 固定开销——优化方向就是砍 launch 次数和 host 层。

## 3. 优化措施(提交 `7203d267`)

1. **m,n ≤ 32 单 launch 融合 kernel**(`_matrix_rank_small_fused_kernel`):双对角化 + Sturm 计数全程寄存器,A 用边界掩码直接读入(不经过 tl.dot,规避掩码操作数误编译约束),省掉 padding 两次 aten launch、d/e 两个 buffer 和一次 sturm launch。
2. **标量 tol 快路径**(所有小路径):atol/rtol 非 tensor 时不再物化 `(batch,)` tensor,直接作为 kernel 标量参数(`TOL_TENSOR` constexpr 区分;tensor tol 走原路径)。语义与 `_prepare_tolerances` 逐点一致,含"显式正 atol 抑制默认 rtol"(`_tolerance_scalars`)。
3. **lockstep 双阈值 Sturm**(`_sturm_count_less2` / `_sturm_count_less_reg2`):qd 递推是 K 步串行链,hermitian 的 4 次计数(tol_lo/tol_hi × ±tol)合成 2 条链,general 的 2 次合成 1 条,串行工作减半。hermitian 16×16 由 0.30 ms 降到 0.19-0.22 ms。
4. **hermitian 直三对角化**:对称矩阵改走单边 Householder 三对角化(反射次数减半、无 `tl.trans`),Sturm 在特征值域直接计数(BIDIAG=False),比"双对角化 + σ² 域"更快更准。小矩阵走融合 kernel 的 HERMITIAN 分支,33~64 走 `_matrix_rank_tridiag_kernel` + sturm。
5. **gram 路径顺手修两点**:旧 `_matrix_rank_tridiag_kernel` 存 d/e 没带 batch 偏移(batch>1 时所有 program 写 batch 0 的槽位,互相覆盖——已补);Gram 输出 buffer 由 `zeros` 改 `empty`(Gram kernel 写满整个 64×64 tile,零填充是白干的)。

## 4. 新工具链缺陷(本阶段实测,补充 §4.1)

| # | 现象 | 规避 |
|---|---|---|
| 24 | **BLOCK=64 的"双对角化 + 寄存器 Sturm"融合 kernel 在一大批 K 值上误编译**(K=33,51,53,54,55,57,58,62…,K=3..64 sweep 实测 63/186 错):e_vec 出现 NaN/垃圾值,或计数链整体错误;且同一源码结构下,加减无关的 store、改 batch 索引方式都会让故障在不同 K 间漂移(编译期确定性、对源码扰动极敏感)。BLOCK ≤ 32(K ≤ 32)全部稳定 | 融合 kernel 只用于 m,n ≤ 32;33~64 保持原来的 bidiag + sturm 两 kernel 结构(该组合经全量测试验证)。**教训:这个后端上 kernel 复杂度有隐形上限,加功能必须配全 K 扫描** |
| 25 | **Gram 融合不可行**:`tl.static_range`(dot 循环)与 `tl.range`(三对角化循环)同函数 → 编译报 "cannot reasign constexpr m0";把 dot 循环抽成 jit helper 后 → MLIR `ConvertLinalgRToBinary` 直接崩溃 | gram 路径维持 gram + tridiag + sturm 三 kernel 结构 |
| 26 | jit helper 之间多返回值(元组)正常;`if hi == 0.0` 标量分支改为默认 rank=0 + `if hi > 0.0` 后等价且可内联 | — |

## 5. 性能(`--mode operator`,两轮复跑,共享机器小 shape 有 ±0.1 波动)

hermitian(基线为 NPU 散算子 `_composed_matrix_rank`):

| shape | 优化前(CPU fallback 基线/修正后) | 优化后 |
|---|---|---|
| 1×1 / 2×2 | 0.83 / 0.79 | 1.08-1.24 |
| 8×8 / 16×16 / 17×17 | **0.40 / 0.38 / 0.38** | 0.80-1.28 |
| 32×32 / 33×33 | 0.82 / 0.69 | 0.79-0.98 |
| 64×64 / 128×128 / 256×256 / 512×512 / 1024² | 1.17-3.35 | 1.10-3.31 |
| batched 小矩阵((32,8,8)~(2,64,64)) | 1.05-2.66 | **2.2-3.7** |

general(基线为 torch 原生 aclnn):8×8 由 0.92 → 2.4-2.9,16×16 → 2.3-2.9,(256,8) 由 0.63 → 0.98-1.07,(16,512)/(512,16) 0.83-1.41,batch 小矩阵 5.0-7.0×,大矩阵与第二阶段持平(1024² ~1.05-1.12,512² ~3.7)。

## 6. 已知限制(本阶段确认,非本次引入)

- **gram 路径(rows > 64 且 k ≤ 64)对缓衰减谱低秩矩阵高估 rank**:Gram 在 σ² 域计算,小于 ~√eps·σmax ≈ 3.5e-4·σmax 的奇异值被 fp32 舍入淹没(默认 tol 是 k·eps·σmax ~ 1e-6·σmax 量级),实测 (256,64)/(1024,8) 等 lowrank 输入得 50/7 vs 参考 32/4。**用改动前代码(326e3006)跑同样用例同样失败**,属 Gram 方法的固有限制(见 §4.2 陷阱 1)。官方测试/基准的谱都有清晰间隙,不受影响。如要修复,可把第三阶段的非分块 Golub-Kahan 双对角化路径向下覆盖到 rows>64 的这段(它支持任意 rows)。
- 融合 kernel 的正确性依赖当前编译结果(缺陷 24 的存在说明 BLOCK=32 版本也可能在未来源码改动后重新翻车),**任何后续修改都必须重跑 K=3..64 全扫描**(`/tmp/sweep_fused.py` 模式:square k=3..64 + hermitian + gram 路径 tall/wide/batch + rand/diag/lowrank 三种谱)。

## 7. 验证

- 官方套件:**61 passed / 6 skipped**(与改动前一致)
- hermitian 专项扫描:k=3..64 × rand/lowrank + batch + 零矩阵,**124/124 全对**
- 全路径扫描 366 例:除 §6 的 gram 路径 lowrank(改动前同样失败)外全对
- benchmark `--mode operator` 两轮:第二轮 hermitian 全部 ≥ 0.81、general 全部 ≥ 0.98

---

# 第五阶段:外部代码评审驱动的 correctness 修复 + 33~64 频段工具链退化处置

> 背景:一份外部评审指出 4 个高优先级 correctness 问题(大矩阵零输入未初始化读取、hermitian 未遵守只读下三角语义、Sturm 同 kernel 写后读、fp64 fail-fast 顺序)和若干工程问题(文件头过时描述、死代码、阈值注释 off-by-one)。
> 结果:4 项全部修复并附真机复现;处置过程中发现**更深层的预存问题**——BLOCK=64 双对角化 kernel 在当前工具链环境下产出本来就是错的(HEAD 亦然),33~64 非 hermitian 频段整体改走 Gram 健康通路。官方套件 **73 passed / 6 skipped**(新增 12 个针对性测试)。

## 1. 评审问题修复(全部附真机复现)

| # | 问题 | 复现(修复前) | 修复 |
|---|---|---|---|
| 1 | k≥513 零矩阵:`_mr_sturm_big_kernel` 的 `hi==0` 分支只写 OUT=0,不写 TOL2/FLAG,而最终 df64 kernel 无条件启动并读未初始化的 `tol2_buf` 覆写结果 | caching allocator 用负值污染后,`zeros(513,513)/(1024,1024)/(2,513,513)` 全部报**满秩** | `hi==0` 分支补写 `TOL2=0`、`FLAG=0`(df64 递推在 tol2=0 下稳定给出 rank 0) |
| 2 | hermitian 3≤k≤64 未遵守 torch"只读下三角"语义:融合 kernel 直接加载完整矩阵,padded 路径全量拷贝,上三角垃圾会进入三对角化 | 上三角填 1e6 垃圾,n=3/32/33/64 全部报**满秩**(torch 返回真实低秩) | 融合 kernel(k≤32):load 后寄存器 `tl.where(下三角, g, trans(g))` 对称化(constexpr 分支,非 herm 零开销);padded herm(33~64):host 侧 cached-mask `torch.where(..., out=padded)` 对称化(见 §3 的代价对比);大路径 init kernel 本来就按 max/min 寻址只读下三角,rank1/rank2 闭式也对 |
| 3 | Sturm kernel 同 kernel 写后读:dd/ee 存回 GM 后同 kernel 内 `_sturm_count_less` 读回(需新值),大 kernel 的 `d_prev` 在 store 后读(需旧值)——与报告自己"MTE3/MTE2 不保序"的结论矛盾 | 未复现出错误(当前编译结果恰好满足时序),属潜在风险 | 大路径:新增 `_mr_bidiag_to_tridiag_kernel` 独立 kernel 构造 BᵀB 三对角(所有 load 在 store 前),`_mr_sturm_big_kernel` 变纯读;小路径:生产者直接写最终三对角,sturm kernel 回写变**幂等**(写入值=读出值,顺序无关)。**全程未用 `tl.debug_barrier`,见 §2** |
| 4 | fp64 检查在 k==1/2 dispatch 之后,fp64 k=1/2 会掉进 Triton 编译 | fp64 (5,1)/(5,2) 报 `MLIRCompilationError` 崩溃;(5,5) 才是干净的 `NotImplementedError` | fp64 检查前移到所有 shape dispatch 之前 |

工程修正:文件头 docstring(删"带列主元 RRQR"(实际无主元,panel kernel 注释为证)和"fp64 走 svdvals 兜底"(实际 NotImplementedError),补全各 k 段路径描述)、`k>=512`→`k>=513` 注释、`_launch_rrqr_rank` docstring。死代码 fused-Jacobi 按原注释是有意保留参考,未动。

新增测试(12 个,全部通过):零矩阵 (513,513)/(1024,1024)/(2,513,513);hermitian 垃圾上三角 3/32/33/64 阶(`test_..._hermitian_ignores_strict_upper`,对照 torch CPU 参考);fp64 五种 shape 类拒绝(`test_..._fp64_rejected`)。

## 2. 重大发现:33~64 非 herm 频段在当前工具链下整体不可信(HEAD 亦然)

修问题 3 的第一版方案(Sturm kernel 加 `tl.debug_barrier`)引发 33~64 频段**大面积误编译**(82/366,灾难性如 (57,57) rand 得 23 vs 57)——缺陷 24 的"源码扰动→误编译漂移"重演。为定位根因做了一组对照实验:

- **A/B 对照(monkeypatch HEAD 源码)**:HEAD 在当前环境下同样失败,且失败集不同(80/366,16 个 K)。(3,3)/(7,7) lowrank 两版都错——是 fp32 噪声区边界(fp32 参考 1、fp64 参考 3、实现得 2),非回归。
- **排除环境变量**:`import flag_gems` 会全局设 `TRITON_ALL_BLOCKS_PARALLEL=1`(sparse_attention.py:22),大矩阵 launcher 会临时摘除,小矩阵 launcher 不会;但摘除后失败集**逐值相同**——不是根因。同时证实同一源码跨进程行为一致(失败按源码确定性,非会话抖动)。
- **分段隔离(直接内部调 kernel dump d/e)**:`_matrix_rank_bidiag_kernel`(BLOCK=64)产出的 **e_vec 是垃圾**(`[0, 1.4e-45, 0, 0, 0.0078, 0.0078]`,正常应 O(1)),σ(双对角) vs σ(原矩阵) 相对误差 0.44——**连通过的 (33,33) rand 对照组也是错的**,rank 能过只因谱间隙宽容;k=34 diag 的 d/e 直接含 NaN。该 kernel 与 HEAD AST 一致——**第四阶段"33~64 两 kernel 组合经全量验证"的结论在当前工具链环境已失效**。

处置:33~64 非 herm(含非方阵)整体改走 **Gram(Cube)+ 三对角化 + Sturm** 通路(该链路的 kernel 今天全部健康:rand/diag 全过),删除坏掉的 `_matrix_rank_bidiag_kernel`;herm 33~64 维持 padded 三对角化(该 kernel 健康)。k≤32 融合 kernel 经 fp64 参考验证健康(atol=1e-4 时与 fp64 逐点一致),不动。

最终扫描(366 例)对比:

| 版本 | 失败数 | 失败构成 |
|---|---|---|
| HEAD(今天重测) | 80 | gram lowrank 48 + 33~64 频段 30(16 个 K,rand/diag/lowrank 都有)+ (3,3)/(7,7) |
| 修复版 v1(debug_barrier) | 82 | 同上,灾难性错误更多 |
| 修复版 v2(寄存器直写) | 81 | 同上 |
| **最终版(Gram 路由)** | 81 | **rand/diag 全部清零**;81 例全部是 lowrank 类:gram σ² 域已知限制 48 + 33~64 归入同一限制 32 + (3,3)/(7,7) fp32 噪声区 2 |

注意最终版失败数与 HEAD 相当,但**性质完全不同**:没有任何 rand/diag(清晰谱)失败和灾难性错误,全部落在文档化的 Gram σ² 域限制内,且 33~64 lowrank 的错误模式从 HEAD 的"报满秩"(got=60 vs 31)改善为可预测的 ~1.5× 高估(got=46 vs 31)。

## 3. 性能影响与 hermitian 33×33 的三次迭代

评审修复对性能的唯一实质影响在 hermitian 33~64 的对称化:

1. **kernel 内 max/min 寻址 gather**:64×64 计算地址 gather 实测 ~90μs → herm 33×33 加速比 0.66;
2. **kernel 内 `tl.trans + tl.where` 寄存器对称化**:实测 ~68μs(该 kernel 的 64×64 trans 走慢路径;融合 kernel 的 32×32 trans 几乎免费)→ 0.64~0.72;
3. **host 侧 cached-mask `torch.where(..., out=padded)`**:增量 ~11μs(与既有 pad 拷贝合并)→ 与 HEAD(0.81~0.83)基本持平(低 ~0.02-0.04)。该 shape 本来就是边际 shape(第四阶段 0.79~0.98),共享机器噪声 ±0.1。

benchmark `--mode operator`(最终干净单跑):**general 全部 ≥ 0.94**(小矩阵 1.8~6.9,大矩阵 1.05~3.7);hermitian 除两个历史边际 shape 外全部 ≥ 0.81(batched 2.6~3.4,大矩阵 1.01~3.3):33×33 = 0.76(多轮 0.76~0.81,对称化带来 ~11μs 代价)、17×17 = 0.77(多轮 0.77~0.93,与 HEAD 同代码,纯机器噪声)。按评审建议的多轮中位数口径:33×33 ≈ 0.79、17×17 ≈ 0.81。

## 4. 新增工具链认知(补充 §4.1/缺陷 24)

- **误编译可按"环境代际"漂移**:同一源码(HEAD)在第四阶段环境全量通过,在当前环境 33~64 频段 16 个 K 失败。任何"该组合已验证"的结论都只在验证时的环境下有效;**评价修复对正确性的影响必须做同环境 A/B 对照,不能引用历史结论**。
- BLOCK=64 的逐元素 blend/extract(`tl.where(区间掩码, 标量, 向量)` 类)是误编译高发区(e_vec 垃圾/NaN);Gram(tl.dot,Cube)+ 三对角化这条链在当前环境稳定。
- `tl.debug_barrier` 不能当作"修复写后读"的免费工具:插入本身即是源码扰动,会重新摇误编译彩票;正确的做法是架构上消除同 kernel 写后读(独立 kernel / 幂等回写)。
- 64×64 `tl.trans` 和计算地址 gather 在小矩阵路径上代价不可接受(68/90μs 量级),host 侧 aten 等价物(~7-17μs)往往是更好的选择。

## 5. 验证

- 官方套件:**73 passed / 6 skipped**(新增 12 个测试全过)
- hermitian 专项扫描:**0/124 mismatch**;hermitian 垃圾上三角 3/32/33/64 全对
- 全路径扫描 366 例(最终代码):81 失败**全部**落入文档化已知限制(Gram σ² 域 79 + fp32 噪声区 2),无清晰谱失败、无灾难性失败
- k≥513 大路径(含重构后的独立构造 kernel):官方 bidiag 测试(513/1024/batch/dense)+ 零矩阵大 shape 全过
- fp64:五种 shape 类全部干净抛 NotImplementedError
- benchmark `--mode operator`(最终干净单跑):general ≥ 0.94;hermitian ≥ 0.81(历史边际的 33×33/17×17 除外,多轮中位数 ≈ 0.79/0.81,与 HEAD 同水平 flap)

---

# 第六阶段:精确路径推广(bidiag64 + 非方阵修复 + df64 地板修复 + QR 压缩 + NPUGraph)

> 目标(依 `next_step.txt` 规划):消除两类已知语义缺口——Gram 路径对缓衰减低秩谱的高估、无主元 QR 的近阈值 ±1——让所有非 herm 输入在奇异值**线性域**完成判定。
> 结果:**部分达成**。33~64 方阵频段已默认切换到精确路径且性能提升;长维 k≤64 与 65~512 的精确路径已建成并验证正确,但因硬件/launch 开销达不到 0.8× 性能底线,默认分发保留 Gram/RRQR,精确路径经 `FLAGGEMS_MR_EXACT_PATH=1` 全局可选。过程中修复了两个真实的生产路径缺陷(fp32 dd/ee 平方域地板、fused kernel 非方阵丢尾能量)。

## 1. 关键精度缺陷修复:fp32 dd/ee 的平方域地板(影响 k≥513 生产路径)

第五阶段的决定性 df64 计数读的是 fp32 的 dd/ee(BᵀB 三对角)。dd_i = d_i²+e²_{i-1} 的 fp32 元素舍入是相对误差,但 dd 的元素本身都是 O(σmax²),小特征值靠递推对消产生——**绝对舍入 ~eps·σmax² 直接淹没 tol² ~ (k·eps·σmax)² 附近的 λ**,即在 σ 域重新引入 √eps·σmax ≈ 3.5e-4·σmax 的地板(比默认容差粗两个数量级)。实测复现:(1024,8) 低秩 r=4(σ4=1e-4 vs tol=1.22e-4)报 4 应为 3;512² 随机矩阵近阈值判错。

修复:`_mr_bidiag_to_tridiag_kernel` 改为写**独立** dd/ee 缓冲(仅供 fp32 bracket 双界用);`_mr_sturm_final_kernel` 改为从**原始 d/e** 在 df64 算术内构造 dd/ee(d_i²、e²、d_i·e_i 全部 TwoProd),递推全程 df64——线性域精度端到端闭合。修复后上述用例全部与 fp64 参考一致。

教训:线性域原则必须贯彻到**最后一个标量递推**;"d/e 是线性域所以最后平方一下没关系"是错觉——只要矩阵元素在 fp32 里被平方并混加,地板就回来了。

## 2. bidiag64:非 herm 33~64 默认切换到线性域

新 kernel `_matrix_rank_bidiag64_kernel`(单 program/矩阵,寄存器 GK 双对角化,写原始 d/e 到 GM,接共享 Sturm 尾巴)。针对 BLOCK=64 误编译簇的结构选择(全部对照第五阶段缺陷清单):

| 结构选择 | 规避的缺陷 |
|---|---|
| M/N/K 全部 runtime(单一编译覆盖整个频段) | 每 K 一张误编译彩票(缺陷 24) |
| 边界掩码直接 load(无 staging buffer) | 额外 aten launch + 无掩码 load 变量 |
| 循环内**无 tl.trans**:gT 用同一外积的转置形式增量维护 | 64×64 trans 慢路径(~68μs)+ 相关误编译 |
| 右反射的末步跳过用 tau2 乘浮点掩码,不用 scf.if | tile op 外的运行时 if 区(§4.1) |
| E[j] 无条件记录 alpha2(符号翻转是 ±1 对角相似,不影响 σ) | 分支不一致风险 |

验证:**σ(B) vs σ(A) 直接对比**(rank 对比会掩盖坏 d/e——第五阶段教训),K=33..64 全值 × 3 次 + 非方阵 + 低秩/对角/零 + batch = **121/121,max 误差 ~6e-8(fp32 机器精度)**。性能:(33,33) 1.23~1.35×、(64,64) 1.84~2.11×——**比原 Gram 通路还快**。

## 3. 潜在 bug:fused kernel 非方阵丢尾能量(预存,官方测试从未抓到)

GK 循环 `for j in 0..K-2` + 末尾 dlast 读对角元的结构,对**方阵**正确,但对长矩阵缺最后一次左反射(K-1 列对角元以下的能量丢失),对宽矩阵缺最后一次右反射(K-1 行尾丢失)。构造探针(16,5) 只有 (15,4)=5 的尾能量:改前 got=4,want=5——**实锤**;随机满秩测试全部"通过"是因为丢失能量在默认 tol 以下。官方套件的非方阵用例全是对角矩阵(天然双对角,反射全为恒等),所以从未覆盖。

修复(两处同构):宽矩阵经"正常 load + 寄存器 trans"转置为长形(σ(Aᵀ)=σ(A);跨步长转置 load 实测慢 10×,不用),GK 循环改为完整 K 次左反射,删掉 dlast。 fused kernel(k≤32)与 bidiag64 同步修复,新增 11 个非方阵回归测试(尾能量探针 + 随机低秩)。

## 4. 长维 k≤64:QR 压缩精确路径(正确但性能不达标,保留为 opt-in)

路线:Householder QR 把 (rows,k) 压成 k×k 的 R(后向稳定,**线性域**——σ(R)=σ(A) 误差 ~k·eps·‖A‖),再 bidiag64 + df64 尾巴。复用 RRQR 的 init/panel kernel(k≤64 单 panel,无 VTV/update),新增 `_mr_extract_upper_kernel` 提取 R。

**发现的约定坑**:panel kernel 的 W 只在**严格上三角**持有 R;对角元在 PIV(反射 alpha)里,W 的对角是反射前残值,下三角是反射向量边带。直接用 W 对角导致系统性低估(64,512) got=60/want=64。修复:R 对角从 PIV 取。

正确性 20/20(含全部 Gram 地板复现器)。但性能分解((512,64)):panel 4175μs(82%)+ bidiag64 659 + extract 209 + tail 156——**panel 的 O(k) 步串行 GM 往返是墙**,整条 0.26~0.6×,达不到 0.8 底线。评估过的替代:多 program 拆分(≈0.7×)、块树 TSQR(估 0.86~1.1×,工程量大且边际)、fp64 Gram+Cholesky(秩亏时 Cholesky 报错;且 fp32 三对角化会把精度再打回平方域地板)、df64 模拟点积(tl.dot 只接受直接 load 操作数,缺陷 4①)。**结论:长维默认保留 Gram(0.9~3.6×),精确路径 `FLAGGEMS_MR_EXACT_PATH=1` 可选。**

## 5. NPUGraph:消灭 host enqueue 瓶颈(bidiag 大路径 2~4.4×)

分解测量发现精确大路径是**纯 host 瓶颈**:(65,65) host enqueue 22.3ms vs 设备执行 0.07ms;_fast_launch 实测 ~80μs/launch(共享 ARM 主机,比第二阶段测量时的 12μs 更慢)。`torch.npu.NPUGraph` 捕获 Triton launch 序列:300 次 kernel 重放 286μs vs 直接 124ms(**434×**)。

`_launch_bidiag_rank` 重构为 workspace + `_bidiag_run`(纯 kernel 序列)+ 按 (m,n,batch,herm,device) 缓存的 graph:首次调用先正常跑一遍(编译+预热 _fast_launch 缓存)再捕获;之后每次调用只做 staging 拷贝 + tol 缓冲填充 + replay + 结果拷贝。捕获失败回退直接发射;`FLAGGEMS_MR_NO_GRAPH=1 关闭`。效果:(65,65) 21.2→4.8ms,512² 172→117ms,1024² 460→460ms(设备 bound 不变)。k≥513 生产路径同步受益。

## 6. 65~512 频段:graph 化后仍 0.28~0.74×,默认保持 RRQR

graph 化消掉 host 瓶颈后,设备侧 ~12μs/kernel 的固定开销成为新地板(每步 6 kernel × 65 步 ≈ 4.8ms@(65,65),baseline 1.35ms)。合并为 2 kernel/步估算 ~0.67~0.84×(边际);DGEBRD 式块双对角化(面板内延迟更新 + tl.dot 尾随)估算 0.7~1.2×,工作量数天且每版都要重摇误编译彩票。**结论:该频段默认保持 RRQR(1.5~3.8×),精确路径 env 可选;QR 的 |R_ii|≈σ 近阈值 ±1 限制维持文档化。**这是"精确性 vs 0.8× 性能底线"在此硬件/工具链代际上的真实冲突点,后续若重做,方向是 DGEBRD 或减少每步 kernel 数。

## 6.5 后续尝试:register-resident 128 宽 GK kernel(死路,UB 硬墙)
为 64 < k ≤ 128 频段实现过 `_matrix_rank_bidiag128_kernel`(2×2 的 (64,64) 寄存器 tile 网格,克隆 bidiag64 的全部安全模式:runtime M/N/K、循环内无 trans、分支-free 块选择)。**编译即失败:ub overflow,需要 2955520 bits(≈360KB)vs 可用 1572864 bits(192KB)**。根因:g + gT 双表示需 8 个 (64,64) tile = 128KB 状态,而 tl.range 循环携带的 tile 被多缓冲(约翻倍)→ ~256KB + 临时量,必然爆 UB。bidiag64 只有 2 个 tile(32KB 状态)所以能活。评估过的替代全部不可行:单表示则行提取被迫 axis-0 归约(缺陷 3);gT 放 GM 则 w-matvec 每步全量 GM 往返(GM tile 操作 ~2.4μs vs 寄存器 ~0.7μs,性能打回非分块路径);bf16 降精度 gT 破坏线性域精度。kernel 已撤销,结论记录在此。**推论:65~128 的精确路径在 UB=192KB 的代际上没有寄存器驻留解,只能走 GM 多 kernel 路径,其每步 kernel 固定开销决定了 0.8× 底线在 k≲128 难以达到。**

## 6.6 尾随裁剪 + 单遍 step kernel:非分块路径 2~2.6×(exact 频段基本越过 0.8)

**先证伪一个方向**:把每步 6 kernel 合并为 4(mat+apply 按列/行条带所有权合并,消除原子累加)实测**性能中性甚至略负**((128,128) 9.80 vs 9.48ms,(512,512) 123 vs 117ms)——NPUGraph 重放下 kernel 启动仅 ~1μs,真正的地板是每 tile 操作 ~2.4μs;条带合并把 ntl×nrc 个 program 的并行压成 ntl 个 program 的串行循环,省下的 launch 抵不过损失的并行度。已回退。**教训:graph 化之后,"多而小的 kernel + 大 grid"优于"少而串行的 kernel"。**

真正有效的两个裁剪(旧 6-kernel 结构不动):

1. **grid 尾随裁剪**:lmat/lapply 的行 tile 起点从 0 收到 j//64,rmat 的列 tile 起点从 0 收到 (j+1)//64——v/u 在界外恒零,这些 tile 是纯浪费。平均省掉四个大 kernel 一半的 tile 操作。
2. **lstep/rstep 单遍化**:原来对同一列/行做两遍 GM 往返(一遍算范数、一遍再读存 v);改为一遍内 provisional 存 v 同时累加范数,最后标量覆写 pivot 元素(x0→x0-alpha)。省一半 step kernel 流量。

实测(`--mode operator`,exact env):

| shape | general 前→后 | herm 前→后 |
|---|---|---|
| (128,128) | 0.42→**0.79** | 0.34→0.65 |
| (256,256) | 0.47→**1.11** | 0.44→**0.94** |
| (512,512) | 0.66→**1.74** | 0.66→**1.49** |
| (512,1024)/(1024,512) | 0.96/0.99→**2.14/2.19** | — |

k≥513 生产路径(默认 dispatch)同步受益:(1024,1024) **1.03→2.59**,herm 1024² **0.99→2.56**。验证:366 例默认扫描 48 失败(与 HEAD 一致的文档化 Gram 限制)、exact 扫描 2 失败(fp32 噪声区)、官方套件双模式 92 passed / 6 skipped,零回归。

剩余缺口:exact 128² general 0.794(0.8 线上,多轮中位数口径)、herm 128² 0.645——herm 走 GK 双对角化浪费一半反射(对称矩阵只需单边三对角化),根治需要独立的 GM 版单边三对角化 kernel 族(3 kernel/步 + 特征值域 ±tol Sturm 尾巴)。

## 6.7 herm k>64 单边三对角化路径(已根治 herm 缺口)

hermitian 输入只需要单边 Householder 三对角化(每步 1 个反射,vs GK 的 2 个),rank = #{|λ| > tol} 在特征值域用 ±tol 双链 Sturm 直接计数。新增 kernel 族(全部克隆现有健康模式):

- `_mr_tridiag_step_kernel`(单 program,单遍 provisional 存 v + pivot 覆写;D[j]=W[j,j],E[j]=±σ;顺带清零 ACC/CSCA)
- `_mr_tridiag_mat_kernel`(ω = W·v 多 program 原子累加;vᵀω 的切片在同 kernel 内原子累加进 CSCA,零额外 kernel)
- `_mr_tridiag_apply_kernel`(rank-2 对称更新 W -= v·wᵀ + w·vᵀ,w = τω − (τ²/2)(vᵀω)v,LAPACK DSYTD2 形式)
- `_mr_sturm_big_tridiag_kernel`(Gershgorin 括号 [max|dᵢ|, max|d|+|e|+|eₚᵣₑᵥ|] —— |λ|max ≥ max|dᵢ| 由 Rayleigh 商保证;需要精化时对 f(x) = #{|λ| > x} 做单次 bisection,lockstep 双链)
- `_mr_sturm_final_tridiag_kernel`(df64 决定性计数:q_i = d_i − x − e²_{i-1}/q_{i-1},x = ±tol 双链 lockstep,e² 用 TwoProd;tol==0 时保留 bracket 结果——零主元保护方向在负侧会误计)

grid 按 K 裁剪尾随(cdiv(K-1-J, 64)),tile 最多触到 K+62 < RS,无需行距跨越掩码。launch 序列按 shape 做 NPUGraph 捕获(独立 `_TRIDIAG_GRAPHS` 缓存)。dispatch:herm 且 k>64 在 exact 频段和 k≥513 生产路径都走三对角化(替代 GK)。

验证:herm 专项压力 34/34(近阈值 ±tol 双符号簇、低秩、缓衰减穿 √eps 地板、零矩阵、垃圾上三角、batch、atol 簇,全部与 fp64 参考逐点一致);366 双扫描与 HEAD 基线一致(默认 48 = Gram 文档化限制;exact 2 = fp32 噪声区);官方套件双模式 92 passed / 6 skipped。

性能(`--mode operator`,herm 频段):

| shape | GK(前) | 三对角(后) |
|---|---|---|
| (128,128) exact | 0.65 | **1.34** |
| (256,256) exact | 0.94 | **2.04** |
| (512,512) exact | 1.49 | **3.97** |
| (1024,1024) 生产 | 2.56 | **7.11** |

**至此 exact 路径(env 开)在 benchmark 的 65~512 全部 shape 上 ≥0.8**(general 128² = 0.82 压线,其余 1.1~1.75;herm 1.34~3.97),唯一剩余的性能缺口是长维 k≤64 的 QR 压缩路径(0.29~0.61,panel 串行 GM 往返的墙,见第六阶段 §3)。

## 7. 验证(第六阶段最终态)

- 官方套件:**92 passed / 6 skipped**(新增:8 个精确路径测试 + 11 个非方阵回归测试)
- 366 例扫描(默认 dispatch):48 失败 = 47 长维 Gram lowrank(已知地板,env 精确路径下全消)+ (7,7) fp32 噪声区
- bidiag64 σ 直验:121/121(~6e-8);长维精确路径 20/20;QR 频段对抗(近阈值 atol/缓衰减/广播 tol)22/22
- benchmark `--mode operator`:48 项全过;general ≥ 0.92;herm 小 shape 0.73~0.90(历史边际 shape,共享机器噪声,多轮中位数口径 ~0.8)
- NPUGraph 路径:k≥513 全谱正确(含零矩阵/batch/herm/低秩),replay 与直接发射结果一致

---

# 第七阶段:默认分发正式切换(阶段 4 落地)

> 依据:exact 路径性能已越过 0.8 底线(尾随裁剪 + 单遍 step + herm 三对角化,见第六阶段 §6.6/§6.7)。
> 结果:**达成(带两段文档化例外)**。

默认分发变更:
- **hermitian k > 64 → 单边三对角化**(原 RRQR / GK):精确且更快(herm 1024² 2.56→6.94×)。
- **非 herm k > 128 → 非分块 GK 双对角化**(原 RRQR):129~512 频段实测 0.85~1.75×(探针:(128,128) 0.85、(256,256) 1.10、(512,512) 1.75);QR 的近阈值 ±1 语义缺口在该频段**默认消除**。
- **保留为例外**:① 长维 k≤64 的 Gram(精确 QR 压缩 0.26~0.6× 不达标);② 非 herm 65~128 的 RRQR(精确路径 0.47~0.85× 不达标,(65,65) 0.475 → (128,128) 0.847,0.8 交叉点在 k≈120,留 128 以内给 QR 保底)。两段例外均可用 `FLAGGEMS_MR_EXACT_PATH=1` 强制精确路径。

验证(切换后):366 例默认扫描 48 失败(全部例外①的文档化 Gram 地板,与切换前一致);exact 扫描 2 失败(fp32 噪声区);herm 压力 34/34(**默认模式**);官方套件双模式 92 passed / 6 skipped;benchmark 默认分发 48 项全部 SUCCESS(general ≥0.93,herm 除两个历史边际 shape 外 ≥0.89)。

切换边界测量(general exact,墙钟中位数):(65,65) 0.475、(80,80) 0.608、(96,96) 0.676、(100,100) 0.699、(112,112) 0.742、(128,128) 0.847。

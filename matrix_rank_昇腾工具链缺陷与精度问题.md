# matrix_rank 昇腾开发:编译器工具链缺陷与精度问题清单

> 本文是 `matrix_rank_昇腾算子实现报告.md` 的专项提炼,聚焦两类问题:
> **(一) 编译器/工具链缺陷** 与 **(二) 精度问题**。所有条目都在昇腾 910B
> 真机上以最小复现确认过,并附规避方法与当前状态。
> 第三部分补充了 exponential 与 cholesky_solve 两个算子在 CANN 9.0.0
> 环境命中的工具链缺陷(整数 mulhi 无可靠通路、bishengir-compile 段错误),
> 定责结论:flagtree 生成的 IR 正确,缺陷均在 CANN bishengir-compile 一侧。
>
> 环境:
> - 原始开发/性能环境:昇腾 910B4(20 AI Core × 2 Vector = 40 Vector 核,
>   UB 192KB),CANN 8.5.0,triton-ascend 3.2.0(BiShengIR),torch 2.6.0+cpu /
>   torch_npu 2.6.0rc1
> - PR CI 兼容环境:CANN 9.1.1,Triton 3.5 frontend + triton-ascend 3.2.1
>   backend(仅第一部分 §1.6 涉及)
>
> 方法论约定:每个可疑模式都写成 ≤20 行的独立 kernel 与 CPU 参考逐元素对比;
> 同一输入连跑 50 次区分"确定性误编译"(恒错,可定位)与"竞态"(结果漂移,
> 状态相关);复杂数学(双对角化等)先在 numpy 上 100 次随机验证再写 kernel。

---

# 一、编译器 / 工具链缺陷

## 1.1 误编译类(编译通过、设备端结果错误或异常)

这是数量最多、代价最高的一类。共性:**与源码扰动极敏感**——加一行无关的
store、改一个 batch 索引方式,故障就在不同 shape/常数之间漂移。因此本仓库的
所有规避写法(下面每条)都是承重墙,不能当作"怪异风格"清理掉。

### 1. 标量 0 常量混入向量的 `tl.where` 误编译

- **现象**:`tl.where(mask, vec, 0.0)`(向量 + 标量常量 0 分支)编译通过,
  设备端概率性出错。早期 rank 错误率:(4,4) 15%、(8,8) 30%、(16,16) 57%。
- **定位**:最小复现矩阵显示错误与掩码内容无关,只与"标量常量 0 作为分支"
  有关;标量分支换成非零表达式(`where(m0, x0-alpha, v)`)则稳定。
- **规避**:清零一律用浮点乘法 `v * mask.to(tl.float32)`,不用 where 的
  标量 0 分支。

### 2. 归约提取的向量参与 `[:,None]` 广播误编译

- **现象**:`v[:,None] * w[None,:]`(v/w 是掩码归约产物)做外积,6/6 全错
  (误差 ~10);同样的写法换纯 load 向量 6/6 全对。
- **规避**:reshape 外积 `tl.reshape(v,(BLOCK,1)) * tl.reshape(w,(1,BLOCK))`
  ——同数据同数学,6/6 全对。**这是本项目最关键的单一发现**(解锁了 22× 的
  rank-2 尾随更新优化)。

### 3. axis-0 掩码归约错误/崩溃

- **现象**:`tl.sum(tl.where(rows == j, g, 0.0), axis=0)`(2D tile 的行轴
  归约)设备端 507035 vector core 异常或结果错误。
- **规避**:行提取全部改为"转置 + axis-1 归约"(`tl.trans` + axis=1)。

### 4. `tl.arange == 常量` 掩码触发设备异常

- **现象**:arange 与常量的等值比较作掩码 → 507035 异常;严格不等式
  (`rows > c`)正常。
- **规避**:等值掩码一律改写为区间掩码 `(x > c-1) & (x < c+1)`。

### 5. 布尔 `tl.sum` 返回错误计数

- **现象**:`tl.sum(bool_mask, axis=0)` 恒返回 1。
- **规避**:先 `.to(tl.int32)` 再 sum。

### 6. 3D 中间体编译失败

- **现象**:`[BP, BLOCK, BLOCK]` 形状的 where/归约(多对并行 Jacobi)编译
  失败,空错误信息。
- **规避**:全部 2D 化。

### 7. 纯动态 `while` 循环携带张量崩溃

- **现象**:`while j < K-1` 携带 tile 的主循环设备端异常;而 `static_range`
  全展开超过 ~32 步编译爆炸(>10min)。
- **规避**:`tl.range` 动态外层 × `tl.static_range` 静态内层 ≤8 步的混合循环。

### 8. BLOCK=64 融合 kernel 的大面积误编译(缺陷彩票)

- **现象**:同一个"双对角化 + 寄存器 Sturm"融合 kernel,在 K=3..64 扫描中
  63/186 个 K 值出错(K=33,51,53,54,55,57,58,62…),表现为 e_vec 出现
  NaN/垃圾或计数链整体错误;**同一源码在第四阶段环境全过、在当前环境 16 个 K
  失败**——误编译可按"环境代际"漂移。BLOCK ≤ 32 全部稳定。
- **规避**:融合 kernel 只用于 m,n ≤ 32;33~64 保持 bidiag + sturm 两 kernel
  结构。
- **教训(本项目最重要的一条)**:该后端上 kernel 复杂度有隐形上限,
  "已验证"的结论只对验证时的环境代际有效;任何源码改动后必须重跑全 K 扫描,
  任何环境切换后历史正确性结论作废,需同环境 A/B 重测。

### 9. 标量 bitcast 误编译

- **现象**:对标量做 `to(tl.int32, bitcast=True)`(用于实现 nextafter 式的
  ULP 步进)结果错误;向量 bitcast 正常。
- **影响与规避**:hermitian 严格阈值需要的"负侧严格 <"不能用
  `nextafter(-tol, -inf)` 实现,改用镜像的零主元 tie 约定(见第二部分 §2.3);
  尺度归一化的 2 的幂掩码必须作用于**向量**(先对行最大值向量掩码,再取
  batch 最大——利用 pow2_floor 在非负浮点上的单调性)。

### 10. `TRITON_ALL_BLOCKS_PARALLEL` 下 atomic 累加误编译

- **现象**:该环境变量(其他模块 import 时全局设置)使大路径的
  matvec/apply kernel 的 atomic_add 求和出错(实测错误和)。
- **规避**:精确大路径编译期间临时 `os.environ.pop` 并在 finally 恢复;
  该变量是进程级全局状态,多线程编译仍可能互相影响,用 `_GRAPH_LOCK` 串行化
  本算子的编译与图捕获。

### 11. 掩码 load + 运行时标量的向量化表达式无法合法化

- **现象**:掩码 load 与任何运行时标量混合的向量表达式,五种写法全部卡在
  MLIR 的 `() -> tensor` 物化上。
- **影响**:tensor 容差的负容差修正不能写 Triton 早退 kernel,只能在 host 侧
  无条件执行一次异步 where(后用归一化的 nonzero flag 替代了全矩阵扫描)。

## 1.2 `tl.dot`(Cube)操作数限制(四重)

| 子缺陷 | 证据 | 规避 |
|---|---|---|
| ① 非 load 直接来源编译错 | 迭代向量喂 dot → "Unsupported op for finding the root alloc" | 只喂直接 load 的 tile |
| ② 输出维 mask/0 填充错 | Gram (5,5) pad 到 (64,64) 后 dot 输出错位 | 输入 pad 到 32 倍数,load 无 mask |
| ③ acc 累加精度错 | 分块 Gram 用 `tl.dot(a,b,g)` 累加,K=64 误差 87~95 | `g = g + tl.dot(a,b)` 独立累加 |
| ④ stride-swapped load 错 | 转置地址装载喂 dot,K=64 Gram 误差 47~50 | `tl.trans(b)` 显式转置(误差 2.3e-5) |

规避后 Gram 精度:K=16/33/64 全部 ≤2.3e-5(对 CPU 双精度参考)。

## 1.3 内存序与同步语义

### 12. MTE3(store)/MTE2(load)是两条独立硬件队列,同 kernel 内不保序

- **现象**:同一 kernel 内 store D/E 再 load 回来,寄存器驻留版 batch=8 实测
  2/8 错;GM 版单独跑正确、被其他算子污染后确定性错;零矩阵 dump 结果漂移
  ——典型竞态。
- **规避(贯穿整个架构)**:分解 kernel 只写、Sturm kernel 只读,**跨 kernel
  边界强制保序**;小路径的回写改为幂等(写入值=读出值,顺序无关)。
- **注意**:`tl.debug_barrier` 不是免费工具——插入本身就是源码扰动,会重摇
  §1.1-8 的误编译彩票(实测曾造成 33~64 频段 82/366 大面积误编译);正确做法
  是架构上消除同 kernel 写后读。

### 13. 跨 program 同步不可靠

- **现象**:atomic 自旋 barrier ≥8 program 时数据不可见(448/512 错),≤4 碰巧
  正确;gather 寻址的同 program GM store→load 误编译,debug_barrier 也救不了。
- **规避**:弃用一切跨 program 同步,同步只走 kernel 边界;主元交换要么物理
  连续 1D 交换,要么(最终方案)直接不交换。

## 1.4 资源 / 可编译性限制

### 14. UB 192KB 硬墙

- (128,128) fp32 tile 不可编译((64,128)/(128,64) 可以);寄存器 tile 一律
  (64,64)。BLOCK=128 全矩阵驻留 GK 直接出局(8 tile 双缓冲 ~360KB > 192KB)。
- **运行时 K 比较混入大 tile 掩码**会让 UB 需求爆炸 25×(6.4M bits);掩码只保留
  编译期可推导的 `lc < B`。
- RRQR 寄存器 panel 的 NB=3(rs=192)是边缘 UB 分配,跨编译在"刚好放得下"和
  "ub overflow"之间反复——launcher 必须把 NB 钳制到 2 的幂 {1,2,4}。

### 15. fp64 kernel 完全不可编译

- rank1/rank2 闭式、fused Jacobi 等所有 fp64 变体都在编译期
  MLIRCompilationError。因此 fp64 输入在所有 shape dispatch 之前 fail-fast
  (`NotImplementedError`),aclnn svd_npu 也仅支持 fp32,无兜底。

### 16. 静态/动态循环结构的组合限制

- `tl.static_range`(dot 循环)与 `tl.range`(三对角化循环)同函数 →
  编译报 "cannot reasign constexpr";抽成 jit helper 后 → MLIR
  `ConvertLinalgRToBinary` 崩溃。Gram 路径维持三 kernel 结构。
- jit helper 之间多返回值(元组)正常;`if hi == 0.0` 标量分支需改写为
  默认值 + `if hi > 0.0` 才可内联。

## 1.5 性能行为类缺陷(不算错,但是隐形天花板)

| # | 现象 | 规避/后果 |
|---|---|---|
| 17 | **JIT dispatch 460µs/launch**(torch native op 仅 31µs) | `jit_fn.warmup()` + `CompiledKernel.run(...)` 预绑定直发(`_fast_launch`,12µs/launch);这是本机所有小 shape 算子的隐形天花板 |
| 18 | 归约类操作极贵:`tl.argmax` 30µs/轮(BK=128),`max+min-where` 18µs,纯 max 11µs | 算法层面去掉选择/主元 |
| 19 | 64×64 `tl.trans` 和计算地址 gather 68/90µs 量级 | host 侧 aten 等价物(~7-17µs)往往更好 |
| 20 | **`do_bench_npu`(KERNEL 模式)对多 kernel 算子无效**:按"一次调用一行"读 profiler csv,把多 kernel 序列记成碎片(基线 1024² 记成 43µs,真实 ~500ms) | 本算子 benchmark 必须用 `--mode operator`(墙钟+sync);散算子基线同理 |

## 1.6 环境/基础设施与工具链代际

- **共享机器抖动**:TBE fork-server EOFError、GE init 失败、他人并发编译时
  bishengir 挂起(共享 /tmp);`PYTHONPATH=src` 覆盖 CANN site-packages 会导致
  `No module named 'tbe'`。规避:PYTHONPATH 保留 CANN 路径、失败重试、清僵尸
  编译进程、benchmark 低值需单独复测再下结论。
- **CANN 8.5.0 → 9.1.1 代际差异(PR CI 实测)**:RRQR 寄存器 panel 的 NB=2/NB=4
  特化在 9.1.1/BiShengIR 下 UB overflow(分别要 8.52M/8.55M bits,可用 1.57M
  bits),`multibuffer=False` 也救不回;最终分发把寄存器 panel 收紧到 rows≤64,
  紧凑 65~255 与全部 herm 输入即使 FAST_PATH=1 也保持精确路径。**同一 kernel
  在 8.5.0 可编译不代表在 9.1.1 可编译。**
- **Triton 3.2 → 3.5 frontend ABI 变化**:低层 launcher 的参数集合从"仅
  non-constexpr"变成"全部已绑定参数",直接调用 `CompiledKernel.run` 的
  `_fast_launch` 在 3.5 下报 "function takes exactly 19 arguments (13 given)";
  按 JITFunction 是否暴露 `non_constexpr_indices` 区分 ABI。
- **triton.knobs(FlagTree/triton ≥3.3)**:合入 master 的通用
  flash_attention_backward 在包导入时 `from triton.knobs import autotuning`,
  本机 triton-ascend 3.2.0 没有该模块,会让 `import flag_gems` 整体失败(与
  本算子无关,本地用未跟踪的 conftest.py shim 绕过)。

---

# 二、开发过程中遇到的精度问题

## 2.1 平方域 vs 线性域(本项目最核心的精度主线)

### 1. Gram 平方的 σ² 域地板

- **问题**:Gram 路径(G=AᵀA 再求特征值)在 σ² 域计算,小于
  ~√(k·eps)·σmax ≈ 3.5e-4·σmax 的奇异值被 fp32 舍入淹没,缓衰减低秩谱的
  rank 被**高估**(实测 (256,64) lowrank 报 50/真值 32;stress 中 σmin 离容差
  64 倍仍判错,因为 σmin² 已小于 Gram 的舍入误差)。
- **结论**:奇异值域的任何平方构造都必须在最后一步用标量完成。方阵 33~64
  默认改走 bidiag64(GK 线性域);长维 Gram 只保留为 `FLAGGEMS_MR_FAST_PATH=1`
  的快速近似,文档化该缺口。

### 2. fp32 dd/ee 把平方域地板带回决定性计数

- **问题**:大路径一度把 BᵀB 三对角(dd_i = d_i²+e²_{i-1})以 fp32 存储再做
  Sturm 计数——dd 的元素舍入是 O(σmax²) 量级,小特征值靠递推对消产生,
  绝对舍入直接淹没 tol² 附近的 λ,等于在 σ 域重新引入 √eps 地板。
- **修复**:决定性计数改用 df64(double-single)算术,直接从原始双对角 d/e
  读取——**df64 救的是计数阶段,救不回 fp32 分解阶段已丢的信息**,这是它的
  适用边界。

### 3. 无主元 QR 的 |R_ii| ≠ σ_i

- **问题**:65~255 快速频段的无主元 Householder QR 不是秩揭示分解,|R_ii|
  系统性偏离 σ_i——缓衰减谱低估 rank,实测普通 randn + 中谱 atol 也低估
  (256² 报 94/真值 129,~-27%)。
- **处置**:这是该快速路径被移出默认分发、仅保留为 opt-in 的根本原因;
  默认全频段走 SVD 级精确路径(GK 双对角化 / 单边三对角化 + df64 Sturm)。

## 2.2 阈值语义(torch 行为对齐)

### 4. 容差公式与严格大于

- torch 语义是 `tol = max(atol, rtol·σmax)`(不是 `atol + rtol·σmax`,早期
  写错);hermitian 的计数是**严格** `#{|λ| > tol} = #{λ > tol} + #{λ < -tol}`。
- Sturm qd 递推的零主元约定(DLANEG:零主元替换为极小负值)天然统计
  `#{λ ≤ x}`,正侧 `K - #{≤ tol}` 恰好严格;**负侧必须用镜像约定**(零主元
  替换为极小正值)才是精确的 `#{λ < -tol}`。
- 教训:`_neg_strict_shift` 用算术位移 `-tol·(1+2eps)` 逼近 `pred(-tol)` 是
  错的——ULP 跳跃随尾数变化(tol=0.5 跳 2 ULP、tol=0.75 跳 3 ULP、最小次正规
  跳 0),会跨过可表示的特征值;且标量 bitcast 路线被缺陷 §1.1-9 堵死。
  最终方案:零主元取正的严格计数 helper,四条 hermitian Sturm 路径统一使用。
- `atol=rtol=0` 的非零缺秩矩阵必须返回真实秩(零谱才返回 0)——两侧阈值都为
  0 时 `K - #{≤0} + #{<0}` 的 tie 约定自然给出正确答案。

### 5. 双负容差(atol<0 且 rtol<0)

- torch 对 tol 不做下钳制:两者皆负时 tol<0,每个 σ≥0 都满足 σ>tol,
  非零矩阵秩=k(零矩阵仍为 0)。kernel 计数公式都假设 tol≥0,因此在返回前
  统一修正;hermitian 的"非零"判定必须只看下三角(垃圾上三角不可见)。
- tensor 容差下不能用 `bool(...any())` 做 Python 分支(每次调用都同步 host),
  修正必须无条件异步执行。

### 6. tensor 容差的边界

- 非 batch 输入的 tensor atol 必须是 **0-D** 张量(shape (1,) 不可广播到
  空 batch shape)——测试构造时实测踩到。
- 0.5-1e-16 这类在 fp32 中不可表示的容差,要求在容差自身精度上比较,
  不能先 round 到 fp32(通用实现要求;昇腾 fp32-only 后端无此能力,
  对应测试在 Ascend 上按 SUPPORT_FP64 跳过)。

## 2.3 分解算法的数学细节

### 7. 双对角化的 LAPACK 约定(三轮调试)

- 左反射含对角(v = B[j:, j]),反射后 D[j] = ±σ;右反射消 j+2 起;
  早期"反射前元素当 d/e"、"次对角多消"、"最后一个对角元丢失"各造成一类
  全错。**CPU 模拟器先行验证数学再移植 kernel 是唯一可行路径**。
- 融合 kernel 曾对**非方阵丢尾能量**:长矩阵缺最后一次左反射、宽矩阵缺最后
  一次右反射——随机满秩测试全部"通过"是因为丢失能量在默认容差之下,官方
  套件的非方阵用例又全是对角矩阵(天然双对角)。构造尾能量探针(16,5) 才
  实锤。教训:覆盖测试必须包含非方阵非对角输入。

### 8. 三对角 apply 的 tau² 溢出

- 强 deflate 谱下尾随次对角 ~1e-10,tau = 2/vnorm² ~ 1e20,`(tau·tau)·(v'w)/2`
  的分组在 fp32 溢出,尾随对角被污染成 -inf/NaN。修复:重新分组为
  `0.5·tau·(tau·cs)`。教训:系数分组顺序是 fp32 下的正确性问题,不只是
  性能问题。

## 2.4 尺度归一化(PR 评审驱动的第十三阶段)

### 9. 极端尺度:溢出/下溢与 1ulp 陷阱

- Householder 代数把矩阵尺度平方(w = A·v 是 O(σ²)):|x|~1e20 时 fp32 平方和
  溢出,|x|~1e-30 时下溢。修复:入口按 per-batch max-abs 归一化,atol 同尺度
  缩放、rtol 不变(rank 在该变换下不变);herm 的尺度只从下三角取。
- **1ulp 陷阱**:最初用普通 max-abs 做除数,~1ulp 的舍入把 366 扫描中
  (24,24) lowrank 的噪声地板奇异值推过默认容差(秩 12 误报 13)。
  修复:除数用位掩码**向下取 2 的幂**(除法是纯指数移位,完全精确,
  对普通输入零舍入代价)。
- 注意 `exp2(floor(log2(x)))` 的 transcendental 路线在该后端实测**不精确**
  (0.5 → 0.49999997),只有位掩码可靠;且位掩码必须作用于向量
  (§1.1-9 的标量 bitcast 缺陷)。

### 10. 残余缺口:单矩阵内部的极端动态范围

- 全局单一 scale 无法同时保存超过 fp32 指数范围的元素:
  `diag([1e20, 1e-30])` 在 `atol=rtol=0` 下,1e-30/2⁶⁶ ≈ 1.4e-50 下溢为 0,
  第二个奇异值不可恢复(torch CPU fp32 LAPACK 对该输入报 2,我们报 1)。
- 这是**与已合入通用实现共享的全局缩放边界**,不是回归(归一化前该输入
  直接 inf/NaN);默认容差下双方一致。完整修复需要 xLASSQ 式安全范数
  (scale/sumsq 对)贯穿所有 kernel,属于后续独立改造。

## 2.5 其他

- **hermitian 下三角语义**:torch 只读下三角。所有路径(load 对称化、host
  where、init kernel 寻址、负容差判定、归一化尺度)都必须只触下三角;上三角
  垃圾值测试是固定回归项。
- **fp32 噪声区**:(3,3)/(7,7) 这类小矩阵缓衰减谱,fp32 参考自身即与 fp64
  不一致——不是后端缺陷,保留在扫描允许清单中。
- **fp64 输入**:fp32-only 后端,所有 shape dispatch 之前 fail-fast(先校验
  dtype 再做尺度归一化——归一化的位掩码是 fp32-only 操作)。

---

# 三、exponential / cholesky_solve 补充缺陷(CANN 9.0.0 环境)

> 本节条目来自另外两个算子的开发实践:**exponential**(指数分布随机采样,
> 命中整数乘高缺陷)与 **cholesky_solve**(Cholesky 求解,命中编译器段错误)。
> 环境:昇腾 910B4,CANN 9.0.0,flagtree 0.7.0+ascend3.5(triton 3.5.1),
> torch 2.10.0+cpu / torch_npu 2.10.0。
> 所有条目均对照 flagtree ttadapter 生成的 Linalg MLIR 定过责:**IR 语义
> 全部正确,缺陷均在 CANN bishengir-compile 一侧**(flagtree 无 lowering
> 错误,仅缺规避逻辑)。

## 3.1 exponential 命中:整数 mulhi 无可靠硬件通路

exponential 用 philox4x32-10 生成均匀随机数(与 torch 随机流对齐),philox
的核心是 32 位无符号乘高(mulhi),实测占 kernel 纯设备时间约 85%(1G fp32:
philox ~80ms / log 变换 ~10ms / 存储 ~5ms)。下面三条替代路径全部被工具链
堵死,只能维持 16 位分段软件实现(~15 条整型指令/mulhi),这是该算子大
shape 只有原生 1/3.6 速度的直接原因。

### 1. `tl.umulhi` 被编译成有符号 mulhi,且慢 60 倍

- **现象**:`tl.umulhi(a, b)` 编译通过但返回**有符号**乘高:
  `umulhi(0xffffffff, 0xffffffff)` 返回 0(正确值 0xfffffffe),
  `umulhi(1, 0xffffffff)` 返回 0xffffffff(正确值 0)——恰为有符号
  (-1)×(-1) 的高位。加符号修正(`hi += (a<0 ? b : 0) + (b<0 ? a : 0)`,
  1M 随机值验证全对)后结果正确,但**性能从 85ms 劣化到 5408ms(60 倍)**,
  lowering 走了灾难性慢路径。
- **定责**:dump ttadapter 输出,flagtree 正确生成 `arith.mului_extended`
  (无符号扩展乘);错误的符号语义与慢路径均在 CANN bishengir-compile 一侧。
- **规避**:维持 `exponential.py` 的 `_mulhi_u32_limb` 16 位分段实现
  (4 次 32 位乘 + 移位/加法)。

### 2. uint64 乘法高位同样按有符号执行,同样慢 60 倍

- **现象**:作为 mulhi 替代路径,`(a.to(u64) * b.to(u64)) >> 32` 的结果
  仍是有符号乘高的值,且同样慢 60 倍。
- **定责**:flagtree 生成的 IR 完全正确(`arith.extui` + `arith.muli i64`
  + `arith.shrui`);CANN 对 i64 muli 的 lowering 符号错误且极慢。
- **注意(Triton 语言规范,非缺陷)**:整型扩展按**源类型**符号性决定
  (`semantic.py` 中 `sign_extend = src.is_int_signed()`),int32→uint64 是
  符号扩展;要零扩展必须先 bitcast 到 uint32 再转 uint64。
- **结论**:与 §1.4-15 的 fp64 不可用同一代际特征——**昇腾后端 64 位
  整数乘同样不可用**。

### 3. int32→f64 向量 cast 不支持(编译失败)

- **现象**:第三条 mulhi 路径(f64 尾数 53 位可精确容纳 16 位分段乘的
  全部中间值)编译即失败:`'hivm.hir.vcast' op currently don't support cast
  int32_t_to_double_rintmode`;uint32→f64 同样不支持。
- **结论**:整数→f64 的任何向量通路都不存在,hivm vcast 只覆盖有限类型对。

### 4. philox 是吞吐瓶颈,ILP 与调参均无效(性能行为类)

- 双流 ILP(两条独立 philox 链交错隐藏串行依赖)反而**慢 2 倍**(N=4096:
  26.7µs vs 14.5µs;1G: 166ms vs 85ms)——瓶颈是整型向量指令吞吐,不是
  延迟,加并行流只增加指令数。
- 配置空间已扫完:BLOCK∈{128..1024} × UNROLL∈{1,2,4} × num_warps∈{1..16}
  在大 N 下全部等价(85.14~85.15ms),**num_warps 在该后端无实际影响**;
  BLOCK=2048 触发 UB overflow。
- 小 N(4096)有 ~16.4µs 固定设备开销(同签名空 kernel 仅 0.8µs),grid
  从 1 加到 32 无变化——philox 串行链的单核延迟地板。
- **残余方向(未采用)**:减少 philox 轮数会破坏与 torch 的随机流一致性
  (`philox_backend_seed_offset` 语义),属语义变更,需 maintainer 决策。

### 5. uint32 逻辑右移被编译成算术右移(符号填充高位)

- **现象**:uint32 张量的 `>>` 全部按**算术**右移执行——符号位置 1 的
  输入高位被 0xffff 填充。最小复现:`0x80000000 >> 16` 得 0xffff8000
  (应为 0x00008000),`0xffffffff >> 16` 得 0xffffffff(应为 0x0000ffff);
  符号位为 0 的输入结果正确;int32 的算术右移本身正常。
- **命中场景**:`exponential_昇腾性能优化绕行方案.md` 的合并 mulhilo
  候选按数学推导删掉了 limb 分解里"冗余"的 `& 0xFFFF` 掩码,CPU(numpy)
  逐位验证全对,但设备结果错——numpy 的 uint32 右移是正常逻辑移位,
  这类后端误编译只有上设备才能暴露。
- **规避**:**uint32 右移后必须紧跟 `&` 掩码清高位**。生产
  `_mulhi_u32_limb` 里每个 `>>` 后的 `& 0xFFFF` 不是冗余,是承重墙,
  不得删除。
- **教训**:所有"逻辑上冗余"的整型掩码/截断,在该后端都应默认视为
  对移位/扩展类 lowering 缺陷的补偿,删除前必须设备重验。

## 3.2 cholesky_solve 命中:bishengir-compile 段错误

### 6. `--enable-auto-bind-sub-block=True` 下 bishengir-compile SIGSEGV

- **现象**:`cholesky_solve_ub_blocked_kernel` 在 FlagGems 代码更新后
  (kernel IR 中 `tl.sum` 从内联变为 `triton/spec/ascend/language/standard.py`
  的显式调用)必现编译期段错误:
  `subprocess.CalledProcessError: ... bishengir-compile ... died with
  <Signals.SIGSEGV: 11>`,stage 为 ConvertLinalgIRToBinary。
- **定位**:dump 出的 kernel MLIR 独立喂给 `bishengir-compile` 稳定复现;
  同一 MLIR 仅把 `--enable-auto-bind-sub-block` 改为 False 即编译通过——
  触发点是编译器内部的子块绑定 pass,不是 IR 本身非法。
- **规避**:launch 时传 `enable_auto_bind_sub_block=False`,并按 PR 评审
  意见用 `torch_npu.utils.get_cann_version(module="CANN")` 做版本门控
  (仅 CANN 9.0.0 禁用,其他版本保持编译器默认;检测失败保守禁用)。
  实测性能影响可忽略((64,8): 0.198ms vs 0.190ms)。
- **教训**:与 §1.6 的代际结论一致——任何环境/代码代际变化后编译稳定性
  结论都要重测;编译器 workaround 参数一律带 CANN 版本判断,不要把
  workaround 无差别地施加到未受影响的环境。

## 3.3 测量口径备忘(重要)

- `do_bench_npu`(KERNEL 模式)测的是**纯设备 kernel 时间**(mspti/
  profiler 抓取),host 开销完全不进指标;`--mode operator` 是墙钟+sync。
  两者结论可以相反:exponential 在 kernel 模式 speedup 0.36(philox 设备
  时间的真实差距),operator 模式 0.97(原生 aten 的 host 调度开销计入后,
  gems 单 kernel 形态反而占优,大 shape fp32 达 1.6~1.75×)。**报性能数据
  必须注明口径**;两个模式的 speedup 差异本身就是"host 开销占比"的度量。
- 同 signature 空 kernel 设备时间 ~0.8µs(910B4),可作为 kernel 启动地板
  的基准值。

---

# 附:给别人复用本经验的三条元教训

1. **最小复现 + 确定性二分 + CPU 先行**:先分清"确定性误编译"与"竞态",
   再用 ≤20 行 kernel 钉死模式;复杂数学先在 numpy 验证。
2. **kernel 复杂度有隐形上限且随环境漂移**:任何"已验证"结论只对当时的
   源码+工具链代际有效;改动后重跑全量扫描,换工具链后全部重测。
3. **正确性验收用对抗谱,不用随机满秩谱**:丢尾能量、阈值 tie、尺度下溢
   这类缺陷在 randn 满秩输入下全部隐形;必须覆盖低秩/缓衰减/近阈值/垃圾
   三角/零矩阵/极端尺度。

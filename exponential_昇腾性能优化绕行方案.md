# exponential 昇腾性能绕行调查

调查对象：本仓库的 out-of-place `exponential`。

> 最新状态：第一轮候选的设备失败结果见 §8；第二轮复查与后续实验方向见 §9。
> §1–7 保留首次调查记录，不能将其中的 CPU 验证或初始建议视为设备优化结论。

本文的 0.3×、Philox 占比约 85% 和 CANN 缺陷数据来自用户描述及仓库的
`matrix_rank_昇腾工具链缺陷与精度问题.md`，不是本轮重新测出的设备数据。

结论：还有保持 Philox4x32-10 算术结果不变的候选路径。优先尝试精简整数
mulhilo；如果仍受整数向量运算限制，实验 FP32 小整数分段乘法。二者均已做
CPU 逐位验证，尚未在 Ascend 上编译或测量，不能承诺设备加速。

## 1. 现状核对

- `_ascend/ops/exponential.py:20` 用四个 16-bit 部分积计算 high32。
  第 38、40 行又各做一次完整的 low32 乘法。每轮有两组乘积，共十轮。
- `_fast_launch` 优化 host 调度，不能解释或解决 kernel 模式下的差距。
- in-place `exponential_.py` 仍直接用 `tl.philox`；本次分析不要混到该文件。
- `tests/test_exponential.py` 测 shape/dtype、输入不变、正值、均值及方差，
  没有覆盖 Philox 位级输出或 generator 状态推进。现有测试全过不能证明随机流保持。
- `philox_backend_seed_offset` 负责状态读写，并不独自保证与原生 torch 的
  counter 布局、uniform 转换和输出排列一致。候选首先保持当前 FlagGems 实现。

## 2. 优先级一：一次生成乘积高低位

见 `candidates.py:mulhilo_limb16`。原高位算法的 `w0` 和 `w1` 已包含低位：

```python
hi = a1 * b1 + (t >> 16) + (w1 >> 16)
lo = (w1 << 16) | (w0 & 0xffff)
```

这把每组乘积从五次整数乘法变为四次，增加低位重建所需的移位/掩码/OR。
同时将输入显式钉为 uint32，移除 `a >> 16`、`w0 >> 16`、`t >> 16`、
`w1 >> 16` 后多余的 `& 0xffff`，让 B 成为显式 constexpr。

`t` 和 `w1` 对任意 uint32 输入都不超出 uint32；右移是逻辑右移。
算法保持 high32/low32 精确，不需要比较、64 位乘法或浮点。

注意：乘法少 20% 不等于 Philox 快 20%。编译器可能已删掉冗余掩码，
而低位重建也有开销。必须同时测试：原实现、只删掩码、合并 mulhilo。
最终以 LLVM/设备指令及 kernel 时间判断，不能数 Python 行数判断。

## 3. 优先级二：FP32 精确整数分段乘法

见 `candidates.py:mulhilo_fp32_11`。设 R=2^11：

```
a = a0 + a1*R + a2*R^2
b = b0 + b1*R + b2*R^2
```

低两段至多 2047，最高段至多 1023。把卷积部分积及进位用 FP32 计算：

```
p0 = a0*b0
p1 = a1*b0 + a0*b1 + floor(p0/R)
p2 = a2*b0 + a1*b1 + a0*b2 + floor(p1/R)
p3 = a2*b1 + a1*b2
p4 = a2*b2
hi = (p2 >> 10) + (p3 << 1) + (p4 << 12)
lo = (p0 & 2047) | ((p1 & 2047) << 11) | ((p2 & 1023) << 22)
```

每个浮点中间结果都是小于 2^24 的非负整数，因此 FP32 可精确表示；
这里没有用 FP32 近似一个完整 32×32 乘积，也不依赖 FMA 残差恢复或 FP64。
即使 mul/add 不收缩为 FMA，数学精度仍成立。B 的三段全是编译期常量。

这条路径绕开文档中的三种故障入口：`umulhi`、i64 multiply、int→fp64。
它没有减少 Philox 轮数，没有改变计数器/种子，也不更换 RNG。

性能风险很实在：每组有九个浮点部分积及更多 cast/shift，而 uint32↔FP32
转换和 UB 中间体也可能昂贵。它是改变指令组合的备选实验，不能仅凭 FP32
通常更快就宣布成功。先做设备 microbenchmark，通过后再接完整 exponential。
官方确认 A2 支持 float 的 FusedMulAdd，但这不保证该 Triton 版本的 lowering：
[CANN FusedMulAdd](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/composite_compute/FusedMulAdd.md)。

## 4. 两轮前缀特化，可和上述路径组合

当前初值是 `(x, c1, 0, 0)`，其中 c1 对整次调用相同。第一轮后：

```
(k0 ^ c1, 0, high(A*x) ^ k1, low(A*x))
```

所以第二轮的 `A*(k0 ^ c1)` 仍是标量，可移出 task 循环，以正常标量
算术或精确 limb helper 计算。随后保留余下八轮，仍是完整的十轮 Philox。
理想情况下二十组向量乘积变十八组，其中第一轮的零乘积通常已经被常量
折叠；剩余实际收益可能很小。先查 IR 是否仍在广播后重复计算，再决定实现。
不要在 host 预计算和缓存未来随机数，或把与 seed 有关的结果错误地跨调用复用。

## 5. 设备验证顺序

1. 同机、同 CANN、同 flagtree 编译原版和候选。先测两个 Philox 常数的
   mulhilo，逐位比较全部 high/low，包括全 1、符号边界、半字进位边界。
2. 跑完整十轮，比较四个 uint32 输出，再测不同 seed/offset 和连续调用。
3. 接回原 exponential 的 counter、输出排列、uniform/log 变换和 generator
   推进逻辑。保留原 BLOCK 先 A/B，避免同时改变随机数排列。
4. 比较 N=4096、1M、16M 和可容纳的大尺寸，以及 fp16/bf16/fp32。核实
   profiler 每次调用对应的全部设备 task，尤其 native 端可能的多任务序列。
   仓库 `benchmark/test_mode.py` 已有保留 CSV、按调用聚合 task 的范例。
5. 候选胜出后再调 BLOCK/grid。新算术的 UB 占用已改变，旧调参无收益的
   结论不能直接套用，但不能先靠调参掩盖算术变慢。

## 6. 收益边界和不优先做的事

若 speedup=0.3、Philox 占总时间 85%，只将 Philox 加速 s 倍，则估计：

```
new_speedup = 0.3 / (0.15 + 0.85 / s)
```

要超过原生，需要 Philox 加速约 5.67 倍；加速两倍也只有约 0.52×。
该估计仅用于说明幅度，原文的分项时间是粗略数据，需用实测占比重算。
小幅精简是有价值的，但不能据此承诺从 0.3× 变成 >1×。

- 不优先重做文档已经否定的 umulhi 符号修正、u64 multiply、FP64、双流 ILP。
- 不通过减少轮数、改 LCG/xorshift 等 RNG 换取默认路径性能。
- 不把 native 随机生成加 Triton log 的多 kernel 路径按单条 profiler 记录记账。
- 不直接迁移新架构的 PhiloxRandom/SIMT。官方 PhiloxRandom 支持表明确
  A2/A3 不支持，950 才支持；升级 CANN 本身不等于 910B 多出硬件能力：
  [CANN PhiloxRandom](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/random_functions/PhiloxRandom.md)。
- AscendC 自定义实现/后端 lowering 修复是后续方向，但同样需要证实更少的
  实际指令或更好的 UB 复用/调度，不能仅因换语言就认为会更快。

## 7. 本轮验证结果和限制

运行 `python verify_cpu.py`，直接把候选函数 AST 的 Triton cast 替换为 NumPy
cast，执行同一份算术主体；参考为 NumPy uint64 完整乘积。

每个候选验证：

- 3,573,052 个输入/乘数对，high32 和 low32 全部逐位一致；
- 655,360 组完整十轮 Philox counter，包括任意四字 counter、exponential
  的后两字为零初值、五种 seed，四个输出均逐位一致。

这不是穷举全部 2^32 输入，完整性的依据还包括上述整数范围与进位推导。
本机未发现 torch、torch_npu、triton、npu-smi，因此未做设备编译、正确性或
性能验证。FlagGems 的生产源码未修改，候选不能直接当已验证优化合入。

## 附录：候选实现与验证程序

以下代码仅为实验候选，未接入生产算子。将两个 Python 代码块分别保存为同目录下的
`candidates.py` 和 `verify_cpu.py`，安装 NumPy 后运行 `python verify_cpu.py`
即可复现 CPU 算术检查；该检查不需要安装 Triton。

### candidates.py

```python
"""Unbenchmarked, exact-arithmetic candidates; not installed into FlagGems.

Both functions return (high32, low32). Inputs must be uint32 and B constexpr.
The caller must retain all ten Philox rounds and its existing counter layout.
"""

import triton
import triton.language as tl


@triton.jit
def mulhilo_limb16(a, B: tl.constexpr):
    a = a.to(tl.uint32)
    a0 = a & 0xFFFF
    a1 = a >> 16
    b0: tl.constexpr = B & 0xFFFF
    b1: tl.constexpr = B >> 16
    w0 = a0 * b0
    t = a1 * b0 + (w0 >> 16)
    w1 = a0 * b1 + (t & 0xFFFF)
    hi = a1 * b1 + (t >> 16) + (w1 >> 16)
    lo = (w1 << 16) | (w0 & 0xFFFF)
    return hi, lo


@triton.jit
def mulhilo_fp32_11(a, B: tl.constexpr):
    a = a.to(tl.uint32)
    x0 = (a & 2047).to(tl.float32)
    x1 = ((a >> 11) & 2047).to(tl.float32)
    x2 = (a >> 22).to(tl.float32)
    b0: tl.constexpr = float(B & 2047)
    b1: tl.constexpr = float((B >> 11) & 2047)
    b2: tl.constexpr = float(B >> 22)
    # Every product and sum is a nonnegative integer below 2**24.
    # Exactness does not depend on FMA contraction.
    p0 = (x0 * b0).to(tl.uint32)
    p1 = (x1 * b0 + x0 * b1 + (p0 >> 11).to(tl.float32)).to(tl.uint32)
    p2 = (
        x2 * b0 + x1 * b1 + x0 * b2 + (p1 >> 11).to(tl.float32)
    ).to(tl.uint32)
    p3 = (x2 * b1 + x1 * b2).to(tl.uint32)
    p4 = (x2 * b2).to(tl.uint32)
    hi = (p2 >> 10) + (p3 << 1) + (p4 << 12)
    lo = (p0 & 2047) | ((p1 & 2047) << 11) | ((p2 & 1023) << 22)
    return hi, lo

```

### verify_cpu.py

```python
"""Execute candidate bodies using NumPy uint32/float32, without Triton/NPU.

This checks arithmetic, NOT Ascend lowering, compilation, or performance.
Usage: python verify_cpu.py
"""

import ast
import json
from pathlib import Path

import numpy as np


def load_candidates():
    # Execute the actual helper bodies so the CPU checks cannot accidentally
    # validate a separately transcribed version. Replace only Triton casts.
    tree = ast.parse(Path(__file__).with_name("candidates.py").read_text())

    class Translate(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            for arg in node.args.args:
                arg.annotation = None
            return self.generic_visit(node)

        def visit_AnnAssign(self, node):
            return self.visit(ast.Assign(targets=[node.target], value=node.value))

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
                node.func.attr = "astype"
            return node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "tl":
                node.value.id = "np"
            return self.generic_visit(node)

    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(Translate().visit(tree)), "candidates", "exec"), scope)
    return [scope["mulhilo_limb16"], scope["mulhilo_fp32_11"]]


def reference(a, b):
    product = a.astype(np.uint64) * np.uint64(b)
    return (product >> 32).astype(np.uint32), product.astype(np.uint32)


def philox(counter, seed, multiply):
    c0, c1, c2, c3 = [c.copy() for c in counter]
    k0, k1 = seed & 0xFFFFFFFF, seed >> 32
    for _ in range(10):
        hi0, lo0 = multiply(c0, 0xD2511F53)
        hi1, lo1 = multiply(c2, 0xCD9E8D57)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


def main():
    candidates = load_candidates()
    rng = np.random.default_rng(20261012)
    edges = {0, 1, 0xFFFFFFFF}
    for bit in range(32):
        for delta in (-1, 0, 1):
            edges.add(((1 << bit) + delta) & 0xFFFFFFFF)
    ramp = np.arange(65536, dtype=np.uint32)
    cases = [np.array(sorted(edges), dtype=np.uint32)]
    # Exhaust each half-word with representative values of the other half.
    for fixed in (0, 1, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF):
        cases.extend([(ramp << 16) | fixed, ramp | (fixed << 16)])
    cases.append(rng.integers(0, 2**32, 1_000_000, dtype=np.uint32))
    constants = (0xD2511F53, 0xCD9E8D57)
    reports = []
    for candidate in candidates:
        for b in constants:
            for a in cases:
                expected = reference(a, b)
                actual = candidate(a, b)
                for got, want in zip(actual, expected):
                    assert got.dtype == np.uint32
                    np.testing.assert_array_equal(got, want)
        counter = [rng.integers(0, 2**32, 65536, dtype=np.uint32) for _ in range(4)]
        seeds = (0, 1, 0xFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF)
        for seed in seeds:
            for got, want in zip(philox(counter, seed, candidate), philox(counter, seed, reference)):
                np.testing.assert_array_equal(got, want)
        # Also exercise exponential's actual c2=c3=0 initialization.
        counter[2:] = [np.zeros(65536, dtype=np.uint32) for _ in range(2)]
        for seed in seeds:
            for got, want in zip(philox(counter, seed, candidate), philox(counter, seed, reference)):
                np.testing.assert_array_equal(got, want)
        reports.append({
            "candidate": candidate.__name__,
            "mulhilo_pairs_checked": sum(a.size for a in cases) * len(constants),
            "philox_10_round_counters_checked": 65536 * len(seeds) * 2,
            "status": "bitwise_equal",
        })
    print(json.dumps({"validation": "CPU arithmetic only", "results": reports}, indent=2))


if __name__ == "__main__":
    main()

```

### 本轮 CPU 验证输出

```json
{
  "validation": "CPU arithmetic only",
  "results": [
    {
      "candidate": "mulhilo_limb16",
      "mulhilo_pairs_checked": 3573052,
      "philox_10_round_counters_checked": 655360,
      "status": "bitwise_equal"
    },
    {
      "candidate": "mulhilo_fp32_11",
      "mulhilo_pairs_checked": 3573052,
      "philox_10_round_counters_checked": 655360,
      "status": "bitwise_equal"
    }
  ]
}
```

---

## 8. 设备验证结果（CANN 9.0.0 / 昇腾 910B4 补测，2026-10）

在目标设备上按 §5 的顺序完成了编译、逐位正确性与性能验证。环境：CANN
9.0.0，flagtree 0.7.0+ascend3.5（triton 3.5.1），torch 2.10.0 + torch_npu
2.10.0；计时用 `do_bench_npu`（纯设备时间），fp32。

**结论：两个候选都没有优化效果，生产实现维持不变。**

### 8.1 优先一 mulhilo_limb16：掩码删不得，合并后慢 6%

- 按文档原版（删除 `>>` 后的 `& 0xFFFF` 掩码）上设备：**结果错误**。
  根因不是算术推导，而是一个新的后端缺陷——**uint32 逻辑右移被编译成
  算术右移**（`0x80000000 >> 16` 得 0xffff8000，高位符号填充）。CPU 侧
  numpy 验证无法暴露此类误编译。生产 `_mulhi_u32_limb` 里每个 `>>` 后的
  掩码是承重墙，不是冗余。
- 补回掩码的合并版（每个乘积 5 次乘降为 4 次，lo 从 `w0/w1` 重建）：
  与生产实现**逐位一致**（4 组 seed/offset × 1M 元素），但**慢约 6%**：

  | N | 生产实现 | 合并 mulhilo |
  |---|---|---|
  | 4096 | 16.41µs | 17.24µs |
  | 16M | 1583.64µs | 1683.03µs |
  | 1G | 95.07ms | 101.03ms |

  lo 重建的 `shift|or` 与增加的寄存器压力吃掉了省下的那一次乘法。

### 8.2 优先二 mulhilo_fp32_11：修通编译后仍慢 1.6 倍

- 原版编译失败：`uint32→fp32` 的 vcast 不支持（`'hivm.hir.vcast' op
  currently don't support cast uint32_t_to_float_rintmode`，与
  int32→f64 缺失同类）；绕成 `uint32→int32→fp32` 后可编译。
- 另需给 `a >> 22` 补掩码（同 8.1 的右移缺陷）才能保证逐位正确。
- 修通后性能仍然**慢约 1.6 倍**（1G：149.99ms vs 95.04ms；4096：
  24.69µs vs 16.48µs）——5 处 int↔fp32 向量转换的成本与整数乘法相当，
  9 次 fp 乘加 cast 不敌 4 次整型乘。fp32 通路的吞吐优势被转换开销抵消。

### 8.3 第一轮设备验证后的判断（后续复查见 §9）

两条候选路径均被设备实测证伪；§4 的两轮前缀特化预期收益微小，未再投入。
philox 在该后端的瓶颈是整型向量指令吞吐，仓库内手段已穷尽，剩余空间在
CANN 一侧（`mului_extended`/i64 muli/uint32 shrui 的 lowering 修复）。
本次新发现的 uint32 右移缺陷已补入
`matrix_rank_昇腾工具链缺陷与精度问题.md` 第三部分（§3.1 条目 5）。

---

## 9. 第二轮复查：测量、XOR、UB 与 DSA（2026-10-12）

依据：已更新的仓库方案 §8、缺陷文档 §3.1，以及当前 exponential 源码。
原合并 mulhilo 慢约 6%、FP32 分段慢约 1.6 倍，均接受设备实测结论。
本轮没有 NPU 实测，未修改生产算子；以下保留既有设备结果，并区分源码线索与待验证假设。

### 9.1 首先复核 0.3× 的分母和分子

发现了三个可以相互印证的源码线索：

1. `FlagGems/benchmark/test_exponential.py` 直接使用 GenericBenchmark；
   `benchmark/base.py:337` 在 kernel 模式直接调用 `do_bench_npu`，没有本算子的
   多任务聚合。
2. 本地 `FlagTree/third_party/ascend/backend/testing.py:245` 的取样公式是
   `func_idx*(num_warmup+num_active)+num_warmup+active_index`，每次调用只取
   一行 `Duration(us)`。当一次调用生成多个 task 时，它取到的是其中一段
   记录，而非完整调用的设备耗时。该 checkout 为 `8ee267ff7`（2026-07-15）；
   **仍须核对目标机器实际安装的 testing.py，不能把本地版本直接当成目标版本。**
3. 官方 op-plugin master 的 `ExponentialKernelNpuOpApi.cpp` 在 950 分支之外，
   依次调用 uniform、neg、add、比较、masked_fill、log、mul。
   这是公开版本证据，不是目标 torch_npu 2.10.0 二进制调用链的实测证明。
   [官方源码](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/ops/opapi/ExponentialKernelNpuOpApi.cpp)

所以：目前不能排除 exponential 和 matrix_rank 命中了同一计时问题。
先保留目标环境 profiler 原始数据，按一次调用聚合全部有效设备任务，明确
AI Vector、DSA、数据搬运是否均被捕获。不要只统计单条 AI Core 记录。
如果 DSA 没出现在当前 profiler 输出中，应补齐采集，不能把未采集计成零。
不要直接切到 operator 模式代替此核查；目标仍是完整调用的设备时间。

计时修正不会让候选比原来的单 kernel 更快；它只会重新确定“离原生还有多远”。
在基线核实前，上一轮按 0.3× 推导出的 5.7 倍目标也只能视为条件估算。

### 9.2 纯 Triton 优先新实验：XOR lowering

`Philox 总耗时占 85%` 不等于 `mulhi 独占 85%`。原始分项是 Philox / log /
store，没有提供 mulhi、普通乘法、XOR、shift/mask 的独立归因数据。
当前每轮四次 XOR，十轮共四十次（优化前源码数）。

官方 A2 Xor 是需要临时空间的高阶接口，支持 int16/uint16，并不是据此就能
假定 Triton 的一个 uint32 `^` 对应一条低成本硬件指令。不过，Triton 究竟
如何展开，必须检查目标 CANN 生成的低层 IR/汇编。
[官方 Xor 文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0591.html)

值得 A/B 的精确恒等式：

```python
def xor(a, b):
    return (a | b) - (a & b)
```

每一位 OR 含有 AND 的所有置位，减法恰好去掉双方均为 1 的位；完整 32 位
输出与 XOR 相同。先只替换 Philox 内的 XOR，保留原乘高、低位乘法和输出转换。
若原 XOR 展开为 AND/OR/NOT/临时复制，可能减少操作或 UB 占用；若后端已经
同样优化，或把该写法重新规范化成 XOR，则没有收益。不能仅数源码指令下结论。

停损条件：最小 kernel 无收益，且完整 Philox 指令/UB 也未减少，就停止此方向。
不要直接重写整个 RNG。

### 9.3 编译调度方向：十轮展开及 UB 生命周期

现有扫参覆盖了 BLOCK、UNROLL、num_warps，但不等于覆盖了循环结构和 UB 分配。
源码用 `tl.static_range(10)` 全展开；BLOCK=2048 的 UB overflow 也值得单独归因。
可保持算术完全不变，对比：

- 十轮全展开；
- `tl.range(0, 10)` 动态十轮；
- 五次动态循环 × 两轮静态展开。

再分别对比 `multibuffer=False`、`enable_ubuf_saving=True`。本地 FlagTree
`third_party/ascend/backend/compiler.py:337` 暴露了这两个选项；以目标环境支持
的参数为准。这种无输入张量、以连续计算为主的 kernel，自动多缓冲不一定获益。

观察 UB 分配、临时复制、同步和每个向量操作的 repeat 利用率；只有确认 UB
下降后才试 BLOCK=2048/4096。动态循环可能增加开销或触发新编译问题，前述
文档已经提示过循环携带张量风险，因此这只是受控实验，不是默认改法。
若不同源码生成相同指令，或硬件向量流水线确实已饱和，则不继续扫。

### 9.4 小幅算术备选：主动使用有符号右移

新右移缺陷说明不能删除既有无符号算法的掩码，但可以换成数学上就需要
算术右移的表达式。见 `round2_candidates.py:mulhi_signed_limbs`：

```
0xD2511F53 - 2^32 = 8019 + (-11695)*2^16
0xCD9E8D57 - 2^32 = -29353 + (-12897)*2^16
```

输入两段仍为 0..65535，用 int32 计算带符号的部分积与进位。
对这两个常数，全部中间值都在 int32 范围内。中间三个 `>>16` 的符号填充
是算法所需，最后加回输入 a 恢复无符号乘高。输入高半字提取处的掩码仍保留。
低位保持原来的单次乘法，避免重走已失败的 shift/or 重建路线。

相对原版大致减少三个中间掩码，增加一次加法，预期只能是小优化，优先级低于
XOR 和完整基线复核。所有有符号/无符号 bitcast 与负常数乘法仍须设备逐位验证。

CPU 验证直接执行候选函数体（替换 Triton cast 为 NumPy view/cast）：

- XOR：2,000,000 组逐位一致；
- signed-limb mulhi：5,572,864 组逐位一致；
- 两者组合的完整 Philox10：262,144 组四字 counter，四个输出全部逐位一致。

命令：`python verify_round2_cpu.py`。此结果只验证算术，不是目标编译器正确性
或性能证据；不能据此合入。

### 9.5 若允许混合后端：DSA uniform + 一个 Triton 变换

本地工作区中的 CANN 参考仓库（不在本仓库内）提供以下硬件路径线索：

- `reference/cann-ops-math/random/dsa_random_uniform/op_host/op_api/aclnn_uniform.cpp:215`
  的 DAV_2201 分支调用 DSARandomUniform；
- 同目录 `dsa_random_uniform.cpp` 使用 `ADD_TO_LAUNCHER_LIST_DSA` 发射，
  并不是普通的 AI Vector Philox kernel。

设备若确认走该分支，可让原生 uniform 使用 DSA，再把 neg/add/边界处理/log/
scale 融合成一个 Triton kernel。这将直接替换最贵的 RNG 阶段，而不是再省几条
整数指令。代价是中间 uniform 张量的 GM 写读及至少两个设备任务。

这不是“纯 Triton Philox 优化”：它需要项目允许使用原生 RNG/混合后端，并重新
界定随机流语义。若要求当前 FlagGems Philox 输出与状态推进逐位不变，不能
直接替换。若目标是对齐 native，则应由原生 generator 推进一次，避免 helper
和 uniform 双重推进；也要处理 FlagGems dispatch 递归。

特别注意 fp16/bf16 的逐步舍入、uniform 端点、正输出处理及 log 的输入精度。
把全部变换直接改为 fp32 一次舍入，不自动等价于原生逐算子低精度路径。
混合路径的性能必须按全部 task 计入，不可只测最后一个 Triton kernel。

### 9.6 建议执行顺序

先核实真实 baseline → XOR 最小实验 → 循环/UB 对照 → signed-limb 小实验。
若允许混合后端，优先单独量 DSA uniform 的完整设备代价，再决定是否实现融合变换。
目前没有证据支持“仓库内所有手段已穷尽”，但也没有证据能保证上述候选突破 1×。

### 9.7 第二轮候选及 CPU 验证程序

将下列两个代码块分别保存为同一目录中的 `round2_candidates.py` 和
`verify_round2_cpu.py`，同时保存前述附录中的 `verify_cpu.py`（本轮复用其中的
精确整数参考函数），安装 NumPy 后运行 `python verify_round2_cpu.py`。
无需 Triton 即可做 CPU 算术检查；设备编译和性能测试仍须另行完成。

**round2_candidates.py**

```python
"""Untested on Ascend: preserve Philox bits, change signed/boolean arithmetic.

Keep the original low32 multiply and ten-round schedule when testing these.
"""

import triton
import triton.language as tl


@triton.jit
def xor_via_or_and_sub(a, b):
    return (a | b) - (a & b)


@triton.jit
def mulhi_signed_limbs(a, B: tl.constexpr):
    tl.static_assert(B == 0xD2511F53 or B == 0xCD9E8D57)
    # a1 remains the NONNEGATIVE upper half-word, despite signed shifts.
    ai = a.to(tl.int32, bitcast=True)
    a0 = ai & 0xFFFF
    a1 = (ai >> 16) & 0xFFFF
    if B == 0xD2511F53:
        b0: tl.constexpr = 8019
        b1: tl.constexpr = -11695
    else:
        b0: tl.constexpr = -29353
        b1: tl.constexpr = -12897
    # B - 2**32 = b0 + 2**16 * b1.
    # All products, t, w1 and hs fit SIGNED int32 for these two constants.
    # These are deliberate arithmetic shifts, not broken unsigned shifts.
    w0 = a0 * b0
    t = a1 * b0 + (w0 >> 16)
    w1 = a0 * b1 + (t & 0xFFFF)
    hs = a1 * b1 + (t >> 16) + (w1 >> 16)
    return hs.to(tl.uint32, bitcast=True) + a
```

**verify_round2_cpu.py**

```python
"""CPU arithmetic check only; no Ascend lowering or timing validation."""

import ast
import json
from pathlib import Path

import numpy as np

from verify_cpu import reference


class Translate(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == 'static_assert'):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            bitcast = any(k.arg == 'bitcast' and k.value.value for k in node.keywords)
            node.func.attr = 'view' if bitcast else 'astype'
            node.keywords = []
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'tl':
            node.value.id = 'np'
        return self.generic_visit(node)


def main():
    path = Path(__file__).with_name('round2_candidates.py')
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    scope = {'np': np}
    exec(compile(ast.fix_missing_locations(Translate().visit(tree)), str(path), 'exec'), scope)
    mulhi = scope['mulhi_signed_limbs']
    xor = scope['xor_via_or_and_sub']
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2**32, 2_000_000, dtype=np.uint32)
    b = rng.integers(0, 2**32, a.size, dtype=np.uint32)
    np.testing.assert_array_equal(xor(a, b), a ^ b)
    ramp = np.arange(65536, dtype=np.uint32)
    inputs = [a]
    for fixed in (0, 1, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF):
        inputs.extend([(ramp << 16) | fixed, ramp | (fixed << 16)])
    for constant in (0xD2511F53, 0xCD9E8D57):
        for values in inputs:
            got = mulhi(values, constant)
            assert got.dtype == np.uint32
            np.testing.assert_array_equal(got, reference(values, constant)[0])
    # Exercise both new helpers together across the full ten-round recurrence.
    for seed in (0, 1, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF):
        original = [rng.integers(0, 2**32, 65536, dtype=np.uint32) for _ in range(4)]
        actual = [x.copy() for x in original]
        k0, k1 = seed & 0xFFFFFFFF, seed >> 32
        for _ in range(10):
            for state, candidate in ((original, False), (actual, True)):
                hi0, lo0 = reference(state[0], 0xD2511F53)
                hi1, lo1 = reference(state[2], 0xCD9E8D57)
                if candidate:
                    hi0 = mulhi(state[0], 0xD2511F53)
                    hi1 = mulhi(state[2], 0xCD9E8D57)
                    n0 = xor(xor(hi1, state[1]), np.uint32(k0))
                    n2 = xor(xor(hi0, state[3]), np.uint32(k1))
                else:
                    n0 = hi1 ^ state[1] ^ k0
                    n2 = hi0 ^ state[3] ^ k1
                state[:] = [n0, lo1, n2, lo0]
            k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
            k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
        for got, want in zip(actual, original):
            np.testing.assert_array_equal(got, want)
    print(json.dumps({
        'validation': 'CPU arithmetic only; no NPU performance claim',
        'xor_pairs': a.size,
        'mulhi_pairs': sum(x.size for x in inputs) * 2,
        'philox_10_round_counters': 65536 * 4,
        'result': 'bitwise_equal',
    }, indent=2))


if __name__ == '__main__':
    main()
```

### 9.8 第二轮 CPU 验证输出

```json
{
  "validation": "CPU arithmetic only; no NPU performance claim",
  "xor_pairs": 2000000,
  "mulhi_pairs": 5572864,
  "philox_10_round_counters": 262144,
  "result": "bitwise_equal"
}
```

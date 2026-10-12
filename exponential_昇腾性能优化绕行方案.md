# exponential 昇腾性能绕行调查

调查对象：本仓库的 out-of-place `exponential`。
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

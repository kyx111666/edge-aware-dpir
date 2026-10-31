# DPIR图像去模糊

基于DPIR的非盲灰度图像去模糊实现。在保持DRUNet及8轮迭代不变的条件下，对数据一致性子问题加入频率加权与空间自适应增量约束，并使用预条件共轭梯度求解。

## 安装

使用Python 3.12，默认在CPU上运行，使用2个计算线程。

```bash
git clone https://github.com/kyx111666/dpir-deblurring-final-project.git
cd dpir-deblurring-final-project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python prepare.py
```

Windows的环境激活命令为`.venv\Scripts\activate`。`requirements-lock.txt`固定复现环境的依赖版本；`requirements.txt`列出直接依赖。

`prepare.py`下载缺失的代码、数据和权重。官方DRUNet灰度权重约131 MB，首次运行需要联网下载。

## 运行

检查求解器并运行单个测试条件：

```bash
python -m unittest -v test_core_solver
python core_experiment.py --stage demo
```

演示使用仓库附带的冻结校准参数，输出至`results/core/demo/`。

重新执行验证选参和全部测试：

```bash
python core_experiment.py --stage all
python check_core_results.py
```

也可分步执行`--stage validate`和`--stage test`。完整实验输出：

- `results/core/validation/`：验证记录和选定参数。
- `results/core/test/metrics.csv`：40个测试条件下，输入及4种复原方法的200行指标。
- `results/core/test/images/`：复原图像、对比图和空间权重图。
- `results/core/test/solver_diagnostics.json`：每轮CG步数和残差。
- `results/core/test/run.json`：实验参数和运行环境。

运行生成的结果不纳入版本控制。重复执行会覆盖相应输出目录。修改核心代码或协议后，须重新运行验证选参。

## 实验协议

Set12的01、02用于验证，03–12用于测试。图像中心裁剪为256×256，使用Levin09中索引0、4的已知模糊核，以及标准差2.55、7.65的高斯噪声（像素范围0–255）。退化使用固定随机种子与循环边界。

对比原DPIR、仅频率加权、均匀增量约束、空间自适应增量约束。四种方法均执行8轮迭代和8次DRUNet调用；空间方法的CG计算另计。评估PSNR、SSIM及Sobel梯度误差。

频率系数固定为β=0.3。空间项约束`D(z−x)`，在验证集的正候选`c ∈ {1, 5, 20}`中选择研究配置，`λ=cσ²`。附带配置为c=1；包含关闭空间项在内的验证推荐为c=0。空间项用于研究其作用，不代表验证集平均画质最优。

## 代码结构

| 文件或目录 | 用途 |
| --- | --- |
| `core_solver.py` | 频率加权、空间边缘权重、增量约束与PCG求解 |
| `core_experiment.py` | 验证选参、4种方法对比及演示入口 |
| `core_protocol.json` | 数据划分、退化条件及求解参数 |
| `experiment.py` | 共用的数据加载、模型加载、原DPIR调度及去噪函数 |
| `test_core_solver.py` | 与独立空间域解、闭式解及原求解器的数值一致性检查 |
| `check_core_results.py` | 完整实验的指标、数据隔离与求解残差复核 |
| `prepare.py`、`sources.json` | 数据、权重下载与来源 |
| `data/` | Set12图像与Levin09模糊核 |
| `upstream/DPIR/` | 固定版本的原作者代码与MIT许可 |
| `results/core/validation/calibration.json` | 当前实现对应的冻结校准参数 |
| `results/frequency_only/validation/calibration.json` | 频率系数β的验证选择依据 |

## 来源

原项目：[cszn/DPIR](https://github.com/cszn/DPIR)。原作者代码与MIT许可保存在`upstream/DPIR/`，官方模型及数据来源见`sources.json`。本实现复用预训练DRUNet，不重新训练网络；数据和模型的权利归原权利人。

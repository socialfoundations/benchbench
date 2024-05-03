<p align="center">
<img src="https://raw.githubusercontent.com/socialfoundations/benchbench/main/assets/benchbench-horizontal.png"  width="66%" />
</p>

**BenchBench** is a Python package that provides a suite of tools to evaluate multi-task benchmarks focusing on
**task diversity** and **sensitivity to irrelevant changes**. 

Research shows that for all multi-task benchmarks there is a trade-off between task diversity and sensitivity. The more diverse a benchmark, the more sensitive its ranking is to irrelevant changes. Irrelevant changes 
are things like introducing weak models, or changing the metric in ways that shouldn't matter.

Please see [our paper](https://github.com/socialfoundations/benchbench) for all relevant background. We applied BenchBench to many existing multi-task benchmarks. Cite as:

```
@inproceedings{zhang2024inherent,
  title={Inherent Trade-Offs between Diversity and Stability in Multi-Task Benchmarks},
  author={Guanhua Zhang and Moritz Hardt},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Quick Start

To install the package, simply run:

```bash
pip install benchbench
```

## Example Usage

To evaluate a cardinal benchmark, you can use the following code:

```python
from benchbench.data import load_cardinal_benchmark
from benchbench.measures.cardinal import get_diversity, get_sensitivity

data, cols = load_cardinal_benchmark('GLUE')
diversity = get_diversity(data, cols)
sensitivity = get_sensitivity(data, cols)
```

To evaluate an ordinal benchmark, you can use the following code:

```python
from benchbench.data import load_ordinal_benchmark
from benchbench.measures.ordinal import get_diversity, get_sensitivity

data, cols = load_ordinal_benchmark('HELM-accuracy')
diversity = get_diversity(data, cols)
sensitivity = get_sensitivity(data, cols)
```

To use your own benchmark, you just need to provide a pandas DataFrame and a list of columns indicating the tasks.
Check the [documentation](https://socialfoundations.github.io/benchbench) for more details.

## Reproduce the Paper

<p align="center">
<img src="https://raw.githubusercontent.com/socialfoundations/benchbench/main/assets/banner.png" height="400" width="600">
</p>

One could check out [cardinal.ipynb](https://githubtocolab.com/socialfoundations/benchbench/blob/main/examples/cardinal.ipynb), [ordinal.ipynb](https://githubtocolab.com/socialfoundations/benchbench/blob/main/examples/ordinal.ipynb) and [banner.ipynb](https://githubtocolab.com/socialfoundations/benchbench/blob/main/examples/banner.ipynb) to reproduce our results using Google Colab with one click.

import numpy as np

from .bbh import load_bbh
from .bigcode import load_bigcode
from .glue import load_glue
from .heim import load_heim
from .helm import load_helm
from .imagenet import load_imagenet
from .mmlu import load_mmlu
from .mteb import load_mteb
from .openllm import load_openllm
from .superglue import load_superglue
from .vtab import load_vtab
from .dummy import load_random_benchmark, load_constant_benchmark
from ..utils.win_rate import WinningRate

cardinal_benchmark_list = [
    "GLUE",
    "SuperGLUE",
    "OpenLLM",
    "MMLU",
    "BigBenchHard",
    "MTEB",
    "VTAB",
]
ordinal_benchmark_list = [
    "BigCode",
    "HELM-accuracy",
    "HELM-bias",
    "HELM-calibration",
    "HELM-fairness",
    "HELM-efficiency",
    "HELM-robustness",
    "HELM-summarization",
    "HELM-toxicity",
    "HEIM-alignment_auto",
    "HEIM-nsfw",
    "HEIM-quality_auto",
    "HEIM-aesthetics_auto",
    "HEIM-alignment_human",
    "HEIM-nudity",
    "HEIM-quality_human",
    "HEIM-aesthetics_human",
    "HEIM-black_out",
    "HEIM-originality",
]


def load_cardinal_benchmark(dataset_name, do_rerank=True, **kwargs):
    """
    Load a cardinal benchmark.

    Args:
        dataset_name(str): Name for the benchmark.
        do_rerank(bool): Whether re-rank the data based on the average score.
        **kwargs: Other arguments.

    Returns:
        tuple:
            pd.DataFrame: data.
            list: cols.
    """
    if dataset_name == "GLUE":
        data, cols = load_glue()
    elif dataset_name == "SuperGLUE":
        data, cols = load_superglue()
    elif dataset_name == "OpenLLM":
        data, cols = load_openllm()
    elif dataset_name == "MMLU":
        data, cols = load_mmlu()
    elif dataset_name == "BigBenchHard":
        data, cols = load_bbh()
    elif dataset_name == "MTEB":
        data, cols = load_mteb()
    elif dataset_name == "VTAB":
        data, cols = load_vtab()
    elif dataset_name == "ImageNet":
        data, cols = load_imagenet(**kwargs)
    elif dataset_name == "Random":
        data, cols = load_random_benchmark(**kwargs)
    elif dataset_name == "Constant":
        data, cols = load_constant_benchmark(**kwargs)
    else:
        raise ValueError

    if do_rerank:
        avg = data[cols].values.mean(1)
        order = sorted(np.arange(len(data)), key=lambda x: -avg[x])
        data = data.iloc[order].reset_index(drop=True)

    return data, cols


def load_ordinal_benchmark(dataset_name, do_rerank=True, **kwargs):
    """
    Load an ordinal benchmark.

    Args:
        dataset_name(str): name for the benchmark
        do_rerank(bool): whether re-rank the data based on the winning rate
        **kwargs: other arguments

    Returns:
        tuple:
            pd.DataFrame: data
            list: cols
    """
    if len(dataset_name.split("-")) == 2:
        dataset_name, subset_name = dataset_name.split("-")
    else:
        subset_name = None

    if dataset_name == "HELM":
        subset_name = "accuracy" if subset_name is None else subset_name
        assert subset_name in [
            "accuracy",
            "bias",
            "calibration",
            "fairness",
            "efficiency",
            "robustness",
            "summarization",
            "toxicity",
        ]
        data, cols = load_helm(subset_name)
    elif dataset_name == "HEIM":
        subset_name = "alignment_human" if subset_name is None else subset_name
        assert subset_name in [
            "alignment_auto",
            "nsfw",
            "quality_auto",
            "aesthetics_auto",
            "alignment_human",
            "nudity",
            "quality_human",
            "aesthetics_human",
            "black_out",
            "originality",
        ]
        data, cols = load_heim(subset_name)
    elif dataset_name == "BigCode":
        data, cols = load_bigcode()
    elif dataset_name == "Random":
        data, cols = load_random_benchmark(**kwargs, num_model=1000)
    elif dataset_name == "Constant":
        data, cols = load_constant_benchmark(**kwargs)
    else:
        raise ValueError

    if do_rerank:
        wr = WinningRate(data, cols)
        win_rate = wr.get_winning_rate()
        order = sorted(np.arange(len(data)), key=lambda x: -win_rate[x])
        data = data.iloc[order].reset_index(drop=True)

    return data, cols

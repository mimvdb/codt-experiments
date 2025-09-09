from pathlib import Path
from codt_py import OptimalDecisionTreeRegressor, OptimalDecisionTreeClassifier
import numpy as np
import pandas as pd
from src.data import get_dataset, DATASETS_REGRESSION, DATASETS_CLASSIFICATION

clustering_lb_0 = {
    "casp": 48.31642303278606,
    "concrete": 1.0621070165840045,
    "energy": 4.338647564675673,
    "fish": 0.6694632370599857,
    "gas": 25.805619738982088,
    "grid": 8.479056747143499,
    "news": 0.27382135363859816,
    "qsar": 0.4674721943944452,
    "query1": 4.7304151290727665,
    "avila": 838.0,
    "bank": 0.0,
    "bean": 0.0,
    "bidding": 0.0,
    "eeg": 0.0,
    "fault": 0.0,
    "htru": 0.0,
    "magic": 0.0,
    "occupancy": 0.0,
    "page": 0.0,
    "raisin": 0.0,
    "rice": 0.0,
    "room": 0.0,
    "segment": 0.0,
    "skin": 0.0,
    "wilt": 0.0,
}
clustering_lb_1 = {
    "casp": 165.03579017050393,
    "concrete": 3.0346514241675355,
    "energy": 13.835137145895967,
    "fish": 1.7192048225519776,
    "gas": 27.997331219068755,
    "grid": 28.369715073208937,
    "news": 0.7704955732227469,
    "qsar": 1.2757648742014913,
    "query1": 9.402310162319239,
    "avila": 2992.0,
    "bank": 0.0,
    "bean": 116.0,
    "bidding": 0.0,
    "eeg": 0.0,
    "fault": 104.0,
    "htru": 0.0,
    "magic": 0.0,
    "occupancy": 0.0,
    "page": 0.0,
    "raisin": 0.0,
    "rice": 0.0,
    "room": 0.0,
    "segment": 62.0,
    "skin": 0.0,
    "wilt": 0.0,
}

def r2_to_mse(y, r2):
    variance = np.sum((y - np.mean(y))**2)
    return variance * (1 - r2)

def acc_to_misclassifications(y, acc):
    return len(y) * (1 - acc)

def info_for_dataset(X, y, dataset):
    unique_feature_values = np.sum(np.apply_along_axis(lambda x: len(np.unique(x)) - 1, axis=0, arr=X))
    unique_y = len(np.unique(y))

    return [dataset, X.shape[0], X.shape[1], unique_feature_values, unique_y]

def lb_for_dataset(df, y, dataset, d0_lb, d1_lb, rec_func):
    record = df[np.logical_and(df["p.dataset"] == dataset, df["p.max_depth"] == 3)]
    timeout = record["p.timeout"].squeeze()
    time = record["o.time"].squeeze()
    timeout_text = ""
    if time >= timeout:
        timeout_text = "*"
    score = record["o.train_score"].squeeze()
    rec_score = rec_func(y, score)

    return [f"{dataset}{timeout_text}", f"{d0_lb/rec_score:.2f}", f"{d1_lb/rec_score:.2f}"]

def info_to_str(arr):
    " & ".join(arr) + " \\\\"

def print_merged(class_info, reg_info):
    for i in range(max(len(class_info), len(reg_info))):
        empty = " & " * (len(class_info[0]) - 1)
        line_class = " & ".join(map(str, class_info[i])) if i < len(class_info) else empty
        line_reg = " & ".join(map(str, reg_info[i])) if i < len(reg_info) else empty
        print(f"{line_class} & {line_reg} \\\\")

def plot_dataset_info(df: pd.DataFrame, output_dir: Path):
    class_info = []
    class_lb = []
    for dataset in DATASETS_CLASSIFICATION:
        X, y = get_dataset(dataset, "classification")
        # d0_lb, d1_lb = OptimalDecisionTreeClassifier(max_depth=3).d0d1_lowerbound(X, y)
        d0_lb, d1_lb = clustering_lb_0[dataset], clustering_lb_1[dataset]
        class_info.append(info_for_dataset(X, y, dataset))
        if class_info[-1][-1] > 2:
            class_lb.append(lb_for_dataset(df, y, dataset, d0_lb, d1_lb, acc_to_misclassifications))
    
    reg_info = []
    reg_lb = []
    for dataset in DATASETS_REGRESSION:
        X, y = get_dataset(dataset, "regression")
        # d0_lb, d1_lb = OptimalDecisionTreeRegressor(max_depth=3).d0d1_lowerbound(X, y)
        d0_lb, d1_lb = clustering_lb_0[dataset], clustering_lb_1[dataset]
        reg_info.append(info_for_dataset(X, y, dataset))
        reg_lb.append(lb_for_dataset(df, y, dataset, d0_lb, d1_lb, r2_to_mse))

    print_merged(class_info, reg_info)
    print("%%%%%%%%%%%%")
    print("%%%%%%%%%%%%")
    print("%%%%%%%%%%%%")
    print("%%%%%%%%%%%%")
    print_merged(class_lb, reg_lb)
    
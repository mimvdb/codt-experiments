from codt_py import OptimalDecisionTreeRegressor, OptimalDecisionTreeClassifier
from src.data import get_dataset, DATASETS_REGRESSION, DATASETS_CLASSIFICATION

def main():
    # for dataset in DATASETS_REGRESSION:
    #     X, y = get_dataset(dataset, "regression")
    #     d0_lb, d1_lb = OptimalDecisionTreeRegressor(max_depth=3).d0d1_lowerbound(X, y)
    #     print(f"{dataset} & {X.shape[0]} & {X.shape[1]} & {d0_lb} & {d1_lb} \\\\")
    for dataset in DATASETS_CLASSIFICATION:
        X, y = get_dataset(dataset, "classification")
        d0_lb, d1_lb = OptimalDecisionTreeClassifier(max_depth=3).d0d1_lowerbound(X, y)
        print(f"{dataset} & {X.shape[0]} & {X.shape[1]} & {d0_lb} & {d1_lb} \\\\")

if __name__ == "__main__":
    main()
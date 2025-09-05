from contextlib import redirect_stdout
from io import StringIO
import numpy as np
from .base import BaseMethod, RunParams
from pycontree import ConTree, Tree
import re

def get_contree_tree(node: Tree):
    if node.is_leaf_node():
        return node.get_label()
    else:
        return [
            node.get_split_feature(),
            node.get_split_threshold(),
            get_contree_tree(node.get_left()),
            get_contree_tree(node.get_right()),
        ]

class ConTreeMethod(BaseMethod):
    def __init__(self, task):
        super().__init__("contree", task)
        assert task == "classification"

    def train_model(self, X, y, params: RunParams):
        model = ConTree(
            max_depth=params.max_depth,
            time_limit=params.timeout,
            verbose=True,
        )

        stdout_io = StringIO()
        with redirect_stdout(stdout_io):
            model.fit(X, y)
        output = stdout_io.getvalue()
        txt_pattern = {
            "calls_d2": (r"Total number of specialized solver calls: (\d+)", int),
            "calls_general": (r"Total number of general solver calls: (\d+)", int),
        }

        matches = {}
        for i in txt_pattern:
            matches[i] = txt_pattern[i][1](
                re.search(txt_pattern[i][0], output, re.M).group(1)
            )

        return (
            model,
            {
                "tree": get_contree_tree(model.get_tree()),
                "expansions": matches["calls_d2"] + matches["calls_general"],
            },
        )

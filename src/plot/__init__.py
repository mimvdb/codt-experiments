from src.plot.ablation import graph_anytime_expansions, graph_anytime_time
from src.plot.debug import all_strats_equal
from src.plot.generalisation import oos_table

PLOT_FUNCS = {
    "all_strats_equal": all_strats_equal,
    "oos_table": oos_table,
    "graph_anytime_expansions": graph_anytime_expansions,
    "graph_anytime_time": graph_anytime_time,
}

from src.setup.ablation import setup_ablation, setup_ablation_branch_relaxation, setup_ablation_ss, setup_ablation_terminal
from src.setup.all_quantbnb import setup_quantbnb_regression, setup_quantbnb_classification
from src.setup.debug import setup_debug
from src.setup.generalisation import setup_generalisation
from src.setup.scalability import setup_scalability

SETUP_FUNCS = {
    "debug": setup_debug,
    "generalisation": setup_generalisation,
    "scalability": setup_scalability,
    "ablation": setup_ablation,
    "ablation_ss": setup_ablation_ss,
    "ablation_terminal": setup_ablation_terminal,
    "ablation_branch_relaxation": setup_ablation_branch_relaxation,
    "quantbnb_regression": setup_quantbnb_regression,
    "quantbnb_classification": setup_quantbnb_classification,
}

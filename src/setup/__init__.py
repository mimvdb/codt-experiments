from src.setup.ablation import setup_ablation
from src.setup.all_quantbnb import setup_quantbnb_regression, setup_quantbnb_classification
from src.setup.debug import setup_debug
from src.setup.generalisation import setup_generalisation
from src.setup.scalability import setup_scalability

SETUP_FUNCS = {
    "debug": setup_debug,
    "generalisation": setup_generalisation,
    "scalability": setup_scalability,
    "ablation": setup_ablation,
    "quantbnb_regression": setup_quantbnb_regression,
    "quantbnb_classification": setup_quantbnb_classification,
}

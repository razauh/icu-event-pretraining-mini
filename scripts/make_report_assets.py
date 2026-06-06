"""Generate README/report tables and figures."""

from icu_pretrain.analysis.plots import make_plots
from icu_pretrain.analysis.tables import make_tables


if __name__ == "__main__":
    make_tables()
    make_plots()


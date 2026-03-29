from .baseline import SVGD
from .curvature import MatrixSVGD, SteinVariationalNewton, SVN
from .localized import KNNSVGD
from .randomized import RBMSVGD, RandomBatchSVGD
from .stochastic import SPOS

__all__ = [
    "KNNSVGD",
    "MatrixSVGD",
    "RBMSVGD",
    "RandomBatchSVGD",
    "SPOS",
    "SVGD",
    "SVN",
    "SteinVariationalNewton",
]

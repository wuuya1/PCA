from .policy import Policy
from .rvoPolicy import RVOPolicy
from .rvoPolicy import RVOPolicy
from .pcaPolicy.pcaPolicy import PCAPolicy
from .pcaPolicy.pca2dPolicy import PCA2DPolicy
from .drvoPolicy import DRVOPolicy
from .orcaPolicy import ORCAPolicy
from .pcaPolicy.rvoDubinsPolicy import RVODubinsPolicy

policy_dict = {
    # rule
    'rvo': RVOPolicy,
    'pca': PCAPolicy,
    'pca2d': PCA2DPolicy,
    'drvo': DRVOPolicy,
    'orca': ORCAPolicy,
    'rvo_dubins': RVODubinsPolicy,
}

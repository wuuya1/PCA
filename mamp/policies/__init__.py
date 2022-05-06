from .policy import Policy
from .rvoPolicy import RVOPolicy
from .rvoPolicy import RVOPolicy
from .pcaPolicy.pcaPolicy import PCAPolicy
from .drvoPolicy import DRVOPolicy
from .orcaPolicy import ORCAPolicy
from .pcaPolicy.rvoDubinsPolicy import RVODubinsPolicy

policy_dict = {
    # rule
    'rvo': RVOPolicy,
    'pca': PCAPolicy,
    'drvo': DRVOPolicy,
    'orca': ORCAPolicy,
    'rvo_dubins': RVODubinsPolicy,
}

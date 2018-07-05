# from .spectral_ordering import SpectralOrdering, SpectralBaseline
from .data import SimilarityMatrix
from .spectral_embedding_ import spectral_embedding
from .utils import evaluate_ordering
from .merge_conn_comp_ import merge_conn_comp

__all__ = ['SimilarityMatrix', 'spectral_embedding', 'evaluate_ordering',
           'merge_conn_comp']
# __all__ = ['SpectralOrdering', 'SimilarityMatrix', 'evaluate_ordering',
#            'inverse_perm', 'make_laplacian_emb', 'SpectralBaseline']

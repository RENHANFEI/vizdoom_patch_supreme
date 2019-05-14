import os
import argparse
from .utils import bcast_json_list
from .utils2 import get_n_feature_maps
from .game_features import parse_game_features

def finalize_args(params):
    """
    Finalize parameters.
    """
    params.n_variables = len(params.game_variables)
    params.n_features = sum(parse_game_features(params.game_features))
    params.n_fm = get_n_feature_maps(params)

    params.variable_dim = bcast_json_list(params.variable_dim, params.n_variables)
    params.bucket_size = bcast_json_list(params.bucket_size, params.n_variables)

    if not hasattr(params, 'use_continuous'):
        params.use_continuous = False

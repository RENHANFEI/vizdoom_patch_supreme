import numpy as np
import cv2

from .labels import get_label_type_id, parse_labels_mapping
import torch
import torch.nn as nn
from torch.autograd import Variable
from logging import getLogger

from .feedforward import DQNFeedforward
from .utils import bool_flag


def process_buffers(game, params):
    """
    Process screen, depth and labels buffers.
    Resize the screen.
    """
    screen_buffer = game._screen_buffer
    depth_buffer = game._depth_buffer
    labels_buffer = game._labels_buffer
    labels = game._labels

    init_shape = screen_buffer.shape[-2:]
    all_buffers = []
    gray = params.gray
    height, width = params.height, params.width

    # screen_buffer
    if game.use_screen_buffer:
        assert screen_buffer is not None
        assert screen_buffer.ndim == 3 and screen_buffer.shape[0] == 3
        if gray:
            screen_buffer = screen_buffer.astype(np.float32).mean(axis=0)
            # resize
            if screen_buffer.shape != (height, width):
                screen_buffer = cv2.resize(
                    screen_buffer,
                    (width, height),
                    interpolation=cv2.INTER_AREA
                )
            screen_buffer = screen_buffer.reshape(1, height, width) \
                                         .astype(np.uint8)
        else:
            # resize
            if screen_buffer.shape != (3, height, width):
                screen_buffer = cv2.resize(
                    screen_buffer.transpose(1, 2, 0),
                    (width, height),
                    interpolation=cv2.INTER_AREA
                ).transpose(2, 0, 1)
        all_buffers.append(screen_buffer)

    # depth buffer
    if game.use_depth_buffer:
        assert depth_buffer is not None
        assert depth_buffer.shape == init_shape
        # resize
        if depth_buffer.shape != (height, width):
            depth_buffer = cv2.resize(
                depth_buffer,
                (width, height),
                interpolation=cv2.INTER_AREA
            )
        all_buffers.append(depth_buffer.reshape(1, height, width))
    else:
        assert depth_buffer is None

    # labels buffer
    if game.use_labels_buffer or game.use_game_features:
        assert not game.use_labels_buffer or game.labels_mapping is not None
        assert labels_buffer is not None and labels is not None
        assert labels_buffer.shape == init_shape

        # split all object labels accross different feature maps
        # enemies / health items / weapons / ammo

        # # naive approach
        # _labels_buffer = np.zeros((max(game.labels_mapping) + 1,)
        #                           + init_shape, dtype=np.uint8)
        # for label in labels:
        #     type_id = get_label_type_id(label)
        #     if type_id is not None:
        #         type_id = game.labels_mapping[type_id]
        #         _labels_buffer[type_id, labels_buffer == label.value] = 255

        # create 4 feature maps, where each value is equal to 255 if the
        # associated pixel is an object of a specific type, 0 otherwise
        _mapping = np.zeros((256,), dtype=np.uint8)
        for label in labels:
            type_id = get_label_type_id(label)
            if type_id is not None:
                _mapping[label.value] = type_id + 1
        # -x is faster than x * 255 and is equivalent for uint8
        __labels_buffer = -(_mapping[labels_buffer] ==
                            np.arange(1, 5)[:, None, None]).astype(np.uint8)

        # evaluate game features based on labels buffer
        if game.use_game_features:
            from vizdoom import GameVariable
            visible = game.game.get_game_variable(GameVariable.USER1)
            assert visible in range(16)
            visible = int(visible)
            game_features = [visible & (1 << i) > 0 for i, x
                             in enumerate(game.game_features) if x]
            if params.dump_freq == 30003:
                label_game_features = [np.any(x) for i, x in enumerate(__labels_buffer) if game.game_features[i + 1]]
                if game.game_features[0]:
                    game_features = [game_features[0]] + label_game_features
                else:
                    game_features = label_game_features
        else:
            game_features = None

        # create the final labels buffer
        if game.use_labels_buffer:
            n_feature_maps = max(x for x in game.labels_mapping
                                 if x is not None) + 1
            if n_feature_maps == 4:
                _labels_buffer = __labels_buffer
            else:
                _labels_buffer = np.zeros((n_feature_maps,) + init_shape,
                                          dtype=np.uint8)
                for i in range(4):
                    j = game.labels_mapping[i]
                    if j is not None:
                        _labels_buffer[j] += __labels_buffer[i]
            # resize
            if init_shape != (height, width):
                _labels_buffer = np.concatenate([
                    cv2.resize(
                        _labels_buffer[i],
                        (width, height),
                        interpolation=cv2.INTER_AREA
                    ).reshape(1, height, width)
                    for i in range(_labels_buffer.shape[0])
                ], axis=0)
            assert _labels_buffer.shape == (n_feature_maps, height, width)
            all_buffers.append(_labels_buffer)

    else:
        assert game.labels_mapping is None
        assert labels_buffer is None
        game_features = None

    # concatenate all buffers
    if len(all_buffers) == 1:
        return all_buffers[0], game_features
    else:
        return np.concatenate(all_buffers, 0), game_features


def get_n_feature_maps(params):
    """
    Return the number of feature maps.
    """
    n = 0
    if params.use_screen_buffer:
        n += 1 if params.gray else 3
    if params.use_depth_buffer:
        n += 1
    labels_mapping = parse_labels_mapping(params.labels_mapping)
    if labels_mapping is not None:
        n += len(set([x for x in labels_mapping if x is not None]))
    return n


models = {
    'dqn_ff': DQNFeedforward,
}

def get_model_class(model_type):
    cls = models.get(model_type)
    if cls is None:
        raise RuntimeError(("unknown model type: '%s'. supported values "
                            "are: %s") % (model_type, ', '.join(models.keys())))
    return cls


def register_model_args(parser, args):
    """
        Parse model parameters.
        """
    # network type
    parser.add_argument("--network_type", type=str, default='dqn_ff',
                        help="Network type (dqn_ff / dqn_rnn)")
    parser.add_argument("--use_bn", type=bool_flag, default=False,
                        help="Use batch normalization in CNN network")
    
    # model parameters
    params, _ = parser.parse_known_args(args)
    network_class = get_model_class(params.network_type)
    network_class.register_args(parser)
    network_class.validate_params(parser.parse_known_args(args)[0])
    
    # parameters common to all models
    parser.add_argument("--clip_delta", type=float, default=1.0,
                        help="Clip delta")
    parser.add_argument("--variable_dim", type=str, default='32',
                        help="Game variables embeddings dimension")
    parser.add_argument("--bucket_size", type=str, default='1',
                        help="Bucket size for game variables")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--update_frequency", type=int, default=4,
                        help="Update frequency (1 for every time)")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="Dropout")
    parser.add_argument("--optimizer", type=str, default="rmsprop,lr=0.0002",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    
    # check common parameters
    params, _ = parser.parse_known_args(args)
    assert params.clip_delta >= 0
    assert params.update_frequency >= 1
    assert 0 <= params.dropout < 1

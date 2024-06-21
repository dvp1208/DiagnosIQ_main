import os
import json
import random
import collections
import datetime
import collections
import numpy as np
from operator import itemgetter
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from torch.distributed import all_reduce
except:
    print('torch.distributed cannot be imported.')


def get_last_model_path(save_dir_path):
    model_paths = []
    checkpoints = sorted([c for c in os.listdir(save_dir_path) if 'ckpt-epoch' in c])

    for ckpt in checkpoints:
        ckpt_epoch = int(ckpt[len("ckpt-epoch="): len("ckpt-epoch=") + 4])
        model_paths.append({
            'ckpt_path': os.path.join(save_dir_path, ckpt),
            'ckpt_epoch': ckpt_epoch,
        })

    model_paths = sorted(model_paths, key=itemgetter('ckpt_epoch'))

    return model_paths[-1]


def config_to_dict(config):
    config = config._asdict()
    new_config = collections.defaultdict(lambda: collections.defaultdict(dict))

    for key, component in config.items():
        component = component._asdict()

        for k, c in component.items():
            new_config[key][k] = c

    return new_config


def overwrite_config(config, key_values: dict):
    config = config._asdict()
    new_config = collections.defaultdict(dict)

    for key, component in config.items():
        component = component._asdict()

        new_setting = {}
        for attribute, setting in component.items():
            if attribute in key_values.keys():
                new_setting[attribute] = key_values[attribute]

            else:
                new_setting[attribute] = setting

        new_config[key] = to_namedtuple(new_setting)

    new_config = to_namedtuple(new_config)

    return new_config


def isinstance_namedtuple(obj):
    return (
        isinstance(obj, tuple) and
        hasattr(obj, '_asdict') and
        hasattr(obj, '_fields')
    )


def to_namedtuple(d):
    return collections.namedtuple('X', d.keys())(*d.values())


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_distributed() -> bool:
    return get_world_size() > 1


def to_cpu(tensor):
    return tensor.detach().cpu()


def minmax_norm(array, vmin=None, vmax=None):
    if vmin is None:
        vmin = array.min()
    if vmax is None:
        vmax = array.max()
    array -= vmin
    array /= (vmax - vmin)
    return array


def as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def load_json(path):
    def _json_object_hook(d):
        for k, v in d.items():
            d[k] = None if v is False else v
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_output_dir_path(config):
    study_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = config.save.study_name + '_' + study_time
    output_dir_path = os.path.join(config.save.output_root_dir, dir_name)
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path


def calc_latent_dim(config):
    return (
        config.dataset.batch_size,
        config.model.z_dim,
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters))),
        int(config.dataset.image_size / (2 ** len(config.model.enc_filters)))
    )


def save_config(config, seed, save_dir_path):
    config_to_save = collections.defaultdict(dict)

    for key, child in config._asdict().items():
        for k, v in child._asdict().items():
            config_to_save[key][k] = v

    config_to_save['seed'] = seed
    config_to_save['save_dir_path'] = save_dir_path

    save_path = os.path.join(save_dir_path, 'config.json')
    os.makedirs(save_dir_path, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_to_save, f)

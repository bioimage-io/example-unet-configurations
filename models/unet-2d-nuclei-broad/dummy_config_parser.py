import os
import importlib
import yaml
from subprocess import call


# for now we allow two different options in python sources:
# local files: ./some/path:PythonName
# library: some.python.library.PythonName
def load_class_from_source(source):
    if ':' in source:
        module, name = source.split(':')
    else:
        name_split = source.split('.')
        module = '.'.join(name_split[:-1])
        name = name_split[-1]
    module = importlib.import_module(module)
    cls = getattr(module, name)
    return cls


def path_from_git(spec):
    delim = ':'
    split = spec.split(delim)
    repo_address = delim.join(split[:2])
    file_path = split[2]
    # TODO make use of this
    # version_or_commit = split[3]
    tmp_folder = './tmp'
    os.makedirs(tmp_folder, exist_ok=True)

    repo_name = os.path.split(repo_address)[1]
    tmp_repo = os.path.join(tmp_folder, repo_name)
    if not os.path.exists(tmp_repo):
        call(['git', 'clone', repo_address, tmp_repo])

    spec_path = os.path.join(tmp_repo, file_path)
    assert os.path.exists(spec_path), spec_path
    return spec_path


# for now we allow two different options for specs:
# local files: ./som/path/spec.object.yaml
# file in git:
def get_spec(spec):
    # this is insufficient in error case handling, but should work for now:
    # we check if the path exists and return it,
    # otherwise, we assume this is
    if os.path.exists(spec):
        return spec
    else:
        spec = path_from_git(spec)
        return spec


# TODO validate the kwargs with the kwargs stored in the spec
def load_class_from_spec(spec, **kwargs):
    spec = get_spec(spec)
    with open(spec, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    source = config['source']
    cls = load_class_from_source(source)
    return cls


def parse_model_specific(config):
    source = config['source']
    kwargs = config.get('kwargs', {})
    model_class = load_class_from_source(source)
    return {'name': model_class, 'kwargs': kwargs}


def parse_train_config(config):

    source = config['source']
    main = load_class_from_source(source)
    kwargs = config.get('kwargs', {})

    setup_config = config['setup']

    reader_conf = setup_config['reader']
    reader_kwargs = reader_conf.get('kwargs', {})
    reader_conf = {'name': load_class_from_spec(reader_conf['spec']), 'kwargs': reader_kwargs}

    sampler_conf = setup_config['sampler']
    sampler_kwargs = sampler_conf.get('kwargs', {})
    sampler_conf = {'name': load_class_from_spec(sampler_conf['spec']), 'kwargs': sampler_kwargs}

    preprocess_conf = setup_config['preprocess']
    preprocess_conf = [{'name': load_class_from_spec(conf['spec']), 'kwargs': conf.get('kwargs', {})}
                       for conf in preprocess_conf]

    loss_conf = setup_config['loss']
    loss_conf = [{'name': load_class_from_spec(conf['spec']), 'kwargs': conf.get('kwargs', {})}
                 for conf in loss_conf]

    optimizer_conf = setup_config['optimizer']
    optimizer_kwargs = optimizer_conf.get('kwargs', {})
    optimizer_conf = {'name': load_class_from_source(optimizer_conf['source']), 'kwargs': optimizer_kwargs}

    return {'reader': reader_conf, 'sampler': sampler_conf, 'preprocess': preprocess_conf,
            'loss': loss_conf, 'optimizer': optimizer_conf, 'main': main, 'kwargs': kwargs}


class GenericConfig:
    def __init__(self, config):
        self.callable_ = config['name']
        self.kwargs = config['kwargs']


class TrainingConfig:
    def __init__(self, config):
        self.reader = GenericConfig(config['reader'])
        self.sampler = GenericConfig(config['sampler'])
        self.preprocess = [GenericConfig(conf) for conf in config['preprocess']]
        self.loss = [GenericConfig(conf) for conf in config['loss']]
        self.optimizer = GenericConfig(config['optimizer'])
        self.main = config['main']
        self.kwargs = config['kwargs']


class ModelConfig:
    def __init__(self, model_config, train_config):
        self.model = GenericConfig(model_config)
        self.training = TrainingConfig(train_config)


# simplified config loader, need to replace it with
# the actual function from 'pybio'
def parse_model_config(config_file):

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # parse the model specific config
    model_config = parse_model_specific(config)

    # parse the training config
    train_config = parse_train_config(config['training'])

    return ModelConfig(model_config, train_config)

import torch
import python_core as pycore
from torch.utils.data import DataLoader
from torch_resource.transformations import apply_transformations
from tqdm import trange


def train(model, sampler, preprocess, loss, optimizer,
          n_iterations=500, batch_size=4, num_workers=2,
          out_file='./weights.pytorch'):

    loader = DataLoader(sampler, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    for ii in trange(n_iterations):
        x, y = next(iter(loader))
        optimizer.zero_grad()

        x, y = apply_transformations(preprocess, x, y)
        out = model(x)
        out, y = apply_transformations(loss[:-1], out, y)
        ll = loss[-1](out, y)

        ll.backward()
        optimizer.step()

    # save model weights
    torch.save(model.state_dict(), out_file)


# TODO mapping of args is imprecise in case of list inputs
def instantiate_from_config(config, *args):
    if isinstance(config, list):
        return [instantiate_from_config(conf, *args) for conf in config]
    obj = config['name'](*args, **config['kwargs'])
    return obj


def train_unet2d_nuclei_broad(model_config_file, **kwargs):
    config = pycore.parse_model_config(model_config_file, for_train=True)

    model = instantiate_from_config(config['model'])

    reader = instantiate_from_config(config['reader'])
    sampler = instantiate_from_config(config['sampler'], reader)

    preprocess = instantiate_from_config(config['preprocess'])
    loss = instantiate_from_config(config['loss'])

    optimizer = instantiate_from_config(config['optimizer'], model.parameters())

    train(model, sampler, preprocess, loss, optimizer, **kwargs)


if __name__ == '__main__':
    train_unet2d_nuclei_broad('./UNet2DNucleiBroad.model.yaml', n_iterations=10)

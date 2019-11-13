from dummy_config_parser import parse_model_config


# TODO use the real parser from pybioimageio
def main(model_config_file):
    config = parse_model_config(model_config_file)
    train_function = config.training.main
    train_kwargs = config.training.kwargs

    print("Call training function ...")
    train_function(config, **train_kwargs)


if __name__ == '__main__':
    main('./UNet2DNucleiBroad.model.yaml')

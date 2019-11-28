from pybio import parse_model_spec


def main(model_spec_path):
    config = parse_model_spec(model_spec_path)
    train_function = config.training.object_
    train_kwargs = config.training.kwargs

    print("Call training function ...")
    train_function(config, **train_kwargs)


if __name__ == "__main__":
    main("./UNet2DNucleiBroad.model.yaml")

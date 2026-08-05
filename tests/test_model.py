import torch

from src.models.model_factory import build_model


def main():

    model = build_model()

    model.eval()

    images = torch.randn(
        4,
        3,
        512,
        512,
    )

    with torch.no_grad():

        prediction = model(images)

    print()

    print(prediction.shape)


if __name__ == "__main__":

    main()
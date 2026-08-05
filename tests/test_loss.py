import torch

from src.training.loss import SegmentationLoss


def main():

    criterion = SegmentationLoss()

    logits = torch.randn(
        4,
        2,
        512,
        512,
    )

    masks = torch.randint(
        0,
        2,
        (
            4,
            512,
            512,
        ),
    )

    loss = criterion(
        logits,
        masks,
    )

    print()

    print(loss)


if __name__ == "__main__":

    main()
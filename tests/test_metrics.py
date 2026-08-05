import torch

from src.training.metrics import SegmentationMetrics


def main():

    metrics = SegmentationMetrics()

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

    results = metrics.calculate(
        logits,
        masks,
    )

    print()

    for key, value in results.items():

        print(
            f"{key:20}: {value:.4f}"
        )


if __name__ == "__main__":

    main()
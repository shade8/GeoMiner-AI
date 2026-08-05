from src.data.dataloader import create_dataloaders


def main():

    train_loader, _, _ = create_dataloaders()

    images, masks = next(iter(train_loader))

    print()

    print(images.shape)

    print(masks.shape)

    print(images.dtype)

    print(masks.dtype)


if __name__ == "__main__":

    main()
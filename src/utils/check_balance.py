from collections import Counter
from src.data.dataloader import get_loaders
from src import config

def check_class_distribution(loader, name="Dataset"):
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.tolist())

    counts = Counter(all_labels)
    print(f"\n📊 {name} Class Distribution:")
    for cls, count in counts.items():
        print(f"  Class {cls}: {count} samples")

def main():
    train_loader, test_loader = get_loaders(
        config.TRAIN_DIR,
        config.TEST_DIR,
        batch_size=config.BATCH_SIZE,
        transform=config.transform
    )

    check_class_distribution(train_loader, "Training")
    check_class_distribution(test_loader, "Testing")

if __name__ == "__main__":
    main()

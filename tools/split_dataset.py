import os
import shutil
import random

def split_dataset(root_dir, output_dir, split_ratio=0.8):
    """
    root_dir: folder containing class subfolders of images
    output_dir: where train/ and test/ should be created
    """
    random.seed(42)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(root_dir):

        # 🚨 IMPORTANT: skip output folders
        if class_name in ["train", "test"]:
            continue

        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        # Only take actual image files
        images = [f for f in os.listdir(class_path)
                  if os.path.isfile(os.path.join(class_path, f))]

        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)

        for img in test_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(test_dir, class_name, img)
            shutil.copy2(src, dst)

        print(f"Class '{class_name}': {len(train_imgs)} train, {len(test_imgs)} test")

    print("\n✔ Dataset split complete! No directory errors 🎉")


if __name__ == "__main__":
    split_dataset(".", ".")

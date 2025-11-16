import os
import shutil

def merge_datasets(source_dirs, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    for src in source_dirs:
        print(f"🔍 Merging from: {src}")

        for class_name in os.listdir(src):
            class_path = os.path.join(src, class_name)

            # skip non-class folders
            if not os.path.isdir(class_path):
                continue

            dest_class_path = os.path.join(dest_dir, class_name)
            os.makedirs(dest_class_path, exist_ok=True)

            for img in os.listdir(class_path):
                src_img = os.path.join(class_path, img)
                dst_img = os.path.join(dest_class_path, img)

                # Prevent overwriting duplicates
                if os.path.exists(dst_img):
                    base, ext = os.path.splitext(img)
                    dst_img = os.path.join(dest_class_path, f"{base}_dup{ext}")

                shutil.copy2(src_img, dst_img)

        print(f"✔ Done merging: {src}")

    print("\n🎉 Combined testing dataset created successfully!")

if __name__ == "__main__":
    merge_datasets(
        [
            "data/set1_masoud/Testing",
            "data/set2_tombacker/Testing",
            "data/set3_sartaj/Testing"
        ],
        "data/combined_testing"
    )


import os

def count_images(folder):
    total = 0

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{class_name}: {count}")
            total += count

    print(f"\nTotal images: {total}")


# 👉 Use this
count_images("dataset/train")
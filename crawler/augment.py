import os
import cv2
import albumentations as A
from albumentations.core.composition import OneOf
from tqdm import tqdm

# Define the augmentation pipeline
transform = A.Compose(
    [
        A.Rotate(limit=40, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.OneOf(
            [
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
            ],
            p=0.5,
        ),
        A.CLAHE(p=0.3),
        A.RandomShadow(p=0.3),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
    ]
)

N = 10


def augment(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            # Create corresponding class folder in output
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # Iterate over images in the class folder
            for image_name in tqdm(
                os.listdir(class_path),
                desc=f"Processing {input_dir} {class_name} images",
                leave=False,
            ):
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Generate N augmented versions
                    for i in range(N):
                        # Apply augmentation
                        augmented = transform(image=image)
                        augmented_image = augmented["image"]

                        # Save augmented image with unique index
                        augmented_image_path = os.path.join(
                            class_output_dir, f"aug_{i}_{image_name}"
                        )
                        cv2.imwrite(augmented_image_path, augmented_image)
                    # Copy original image to output directory
                    original_image_path = os.path.join(class_output_dir, image_name)
                    cv2.imwrite(original_image_path, image)


def main():
    dirs = ["validation", "train"]

    for d in dirs:
        input_dir = f"dataset/{d}"
        output_dir = f"dataset/augmented_{d}"
        augment(input_dir, output_dir)


if __name__ == "__main__":
    main()

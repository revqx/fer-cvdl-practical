from augment import select_augmentations, augment_images

if __name__ == "__main__":
    input_dir = "/Users/marius/github/fer-cvdl-practical/data/augmentations"
    output_dir = "/Users/marius/github/fer-cvdl-practical/data/augmentations"
    augmentations_str = "TrivialAugmentWide"

    augmentations = select_augmentations(augmentations_str)
    augment_images(input_dir, output_dir, augmentations)

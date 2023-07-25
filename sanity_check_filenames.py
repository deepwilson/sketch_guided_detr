import os

def get_image_names_from_text_file(file_path):
    image_names = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # print(parts)
            if len(parts) == 2:
                filename = parts[1]
                if filename.startswith("cocostuff"):
                    name = parts[1].split("_")[-1]
                    image_names.append(name)

    return image_names

def get_image_names_from_directory(directory_path):
    image_names = []
    for filename in os.listdir(directory_path):
        filename = filename.replace(".png", ".jpg")
        image_names.append(filename)

    return image_names

if __name__ == "__main__":
    # Replace these paths with your actual file and directory paths
    text_file_path = "data/sketchyCOCO/image_source/object_image_train_source.txt"
    image_directory_path = "data/sketchyCOCO/Scene/GT/trainInTrain"

    # Step 1: Get image names from the text file
    text_file_image_names = get_image_names_from_text_file(text_file_path)
    # for i in text_file_image_names:
    #     print(i)
    # Step 2: Get image names from the directory
    directory_image_names = get_image_names_from_directory(image_directory_path)

    # Step 3: Compare and find missing image names
    for i,j in zip(directory_image_names, text_file_image_names):
        print(i,j)
    missing_image_names = [name for name in directory_image_names if name not in text_file_image_names]

    # Step 4: Count the number of image names that don't exist in the text file
    num_missing_images = len(missing_image_names)

    print(f"Number of image names that don't exist in the text file: {num_missing_images}")
    print("Missing image names:")

    # for name in missing_image_names:
    #     print(name)

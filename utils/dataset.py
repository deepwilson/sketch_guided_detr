# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# from pycocotools.coco import COCO
# import torchvision.datasets as datasets

# class COCODataset(datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transform=None):
#         super(COCODataset, self).__init__(img_folder, ann_file)
#         self._transforms = transform
    
#     def _load_image(self, id: int) -> Image.Image:
#         path = self.coco.loadImgs(id)[0]["file_name"]
#         return Image.open(os.path.join(self.root, path.replace("jpg", "png"))).convert("RGB")

#     def __getitem__(self, index):
#         img, target = super(COCODataset, self).__getitem__(index)
#         print(f"{target}")
#         image_id = self.ids[index]
#         target = {'image_id': image_id, 'annotations': target}
#         if self._transforms is not None:
#             img = self._transforms(img)
#         return img, target

# if __name__ == "__main__":
#     print("start ---------------------")
#     image_path = "./data/sketchyCOCO/Scene/Sketch/paper_version/trainInTrain"
#     # annotations_path = "./data/sketchyCOCO/Scene/Annotation/paper_version/trainInTrain/BBOX"
#     annotations_path = "coco_dataset.json"

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to a common size
#         transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
#     ])

#     dataset = COCODataset(image_path, annotations_path, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

#     # Example usage of the DataLoader
#     for batch in dataloader:
#         images, targets = batch
#         bboxes = targets['annotations']
#         print(images.shape, bboxes)

import os
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset

from utils import transforms as T
from torch.utils.data.dataloader import DataLoader


class COCODataset(Dataset):
    def __init__(self, root: str, annotation: str, targetHeight: int, targetWidth: int, numClass: int):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())

        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.numClass = numClass

        self.transforms = T.Compose([
            T.RandomOrder([
                T.RandomHorizontalFlip(),
                T.RandomSizeCrop(numClass)
            ]),
            T.Resize((targetHeight, targetWidth)),
            T.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=0),
            T.Normalize()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

        with open('classes.txt', 'w') as f:
            f.write(str(classes))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]

        imgInfo = self.coco.imgs[imgID]
        imgPath = os.path.join(self.root, str(imgInfo['file_name']))

        image = Image.open(imgPath).convert('RGB')
        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        image, targets = self.transforms(image, targets)

        return image, targets

    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []

        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # convert from [tlX, tlY, w, h] to [centerX, centerY, w, h]
            bbox[0] += bbox[2] / 2
            bbox[1] += bbox[3] / 2

            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]

            ans.append(bbox + [cat])

        return np.asarray(ans)


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    # return batch
    return torch.stack(batch[0]), batch[1]



if __name__ == "__main__":
    print("start ---------------------")
    image_path = "./data/sketchyCOCO/Scene/Sketch/paper_version/trainInTrain"
    # annotations_path = "./data/sketchyCOCO/Scene/Annotation/paper_version/trainInTrain/BBOX"
    annotations_path = "./coco_dataset.json"


    dataset = COCODataset(image_path, annotations_path, 512, 512, 90)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collateFunction)

    # Example usage of the DataLoader
    for batch in dataloader:
        images, targets = batch
        bboxes = targets
        print(images.shape, bboxes)

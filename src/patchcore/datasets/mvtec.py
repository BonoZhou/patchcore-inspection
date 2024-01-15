import os
from enum import Enum

import PIL
import torch
from torchvision import transforms
import platform

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=[256,256],
        imagesize=[256,256],
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)


        self.imagesize = (3, imagesize[0], imagesize[1])


    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path, defectpos = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        
        if self.split == DatasetSplit.TEST and mask_path is not None:
            try:
                mask = PIL.Image.open(mask_path)
                mask = self.transform_mask(mask)
            except:
                mask = torch.zeros([1, *image.size()[1:]])    
        else:
            mask = torch.zeros([1, *image.size()[1:]])
        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            "defectpos":defectpos,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        defectpos_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}
            defectpos_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    try:
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    except:
                        maskpaths_per_class[classname][anomaly] = None

                    #Zhou Added: get defect position
                    try:
                        defectpath = os.path.join(os.path.dirname(classpath), "DefectImage.txt")
                        print(defectpath)
                        defectpos_per_image = {}
                        if os.path.isfile(defectpath):
                            with open(defectpath,'r') as f:
                                for line in f:
                                    # 分割每一行的数据
                                    image_path, *defect_pos = line.strip().split(',')

                                    # 将图像路径转换为完整路径
                                    image_name = image_path.split('\\')[-1]

                                    # 将缺陷位置转换为整数
                                    defect_pos = list(map(int, defect_pos))

                                    # 将数据添加到字典中
                                    if image_name in defectpos_per_image:
                                        defectpos_per_image[image_name].append(defect_pos)
                                    else:
                                        defectpos_per_image[image_name] = [defect_pos]
                            
                        #print(defectpos)

                        #print(defectpos)
                        defectpos_per_class[classname][anomaly] = defectpos_per_image
                        
                    except:
                        defectpos_per_class[classname][anomaly] = None
                    
                else:
                    maskpaths_per_class[classname]["good"] = None


        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    #Zhou Added: get defectpos and maskpath
                    if platform.system() == 'Windows':  
                        image_name = image_path.split("\\")[-1]
                    else:
                        image_name = image_path.split("/")[-1]
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        if maskpaths_per_class[classname][anomaly] is not None:
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    else:
                        data_tuple.append(None)
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        if defectpos_per_class[classname][anomaly].get(image_name) is not None:
                            data_tuple.append(defectpos_per_class[classname][anomaly][image_name])
                        else:
                            data_tuple.append([])
                    else:
                            data_tuple.append([])
                    data_to_iterate.append(data_tuple)
        return imgpaths_per_class, data_to_iterate


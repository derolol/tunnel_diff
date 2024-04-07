import os
from typing import Sequence, Dict, Union

import cv2
import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision.transforms import transforms

class TunnelDefectDataset(data.Dataset):
    
    def __init__(
        self,
        file_list,
        image_root,
        annotation_root,
        image_format,
        annotation_format,
        out_size,
        apply_transform
    ) -> "TunnelDefectDataset":
        
        super().__init__()

        self.file_list = self.read_file_list(file_list)
        self.image_root = image_root
        self.annotation_root = annotation_root
        self.image_format = image_format
        self.annotation_format = annotation_format
        self.out_size = (out_size, out_size)
        self.apply_transform = apply_transform

    def read_file_list(self, file_list):
        paths = []
        with open(file_list, 'r') as f:
            for path in f.readlines():
                paths.append(path.strip())
        return paths
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        image_name = self.file_list[index]
        
        image_path = os.path.join(self.image_root, f'{image_name}.{self.image_format}')
        annotation_path = os.path.join(self.annotation_root, f'{image_name}.{self.annotation_format}')
        
        image_pil = Image.open(image_path).convert('RGB')
        low_pil = self.low_fea(np.array(image_pil) / 255)
        annotation_pil = Image.open(annotation_path).convert('L')

        image = transforms.F.to_tensor(image_pil).float()
        low = transforms.F.to_tensor(low_pil).float()
        annotation = transforms.F.pil_to_tensor(annotation_pil).long()

        # Resize
        image = transforms.F.resize(image,
                                    size=self.out_size,
                                    interpolation=transforms.InterpolationMode.BILINEAR)
        low = transforms.F.resize(low,
                                  size=self.out_size,
                                  interpolation=transforms.InterpolationMode.BILINEAR)
        annotation = transforms.F.resize(annotation,
                                         size=self.out_size,
                                         interpolation=transforms.InterpolationMode.NEAREST)
        annotation = annotation.squeeze(dim=0)

        # ResizeCrop
        # i, j, h, w = transforms.RandomResizedCrop.get_params(img=image, scale=[0.5, 2.0], ratio=[3./4., 4./3.])
        # image = transforms.F.resized_crop(image, top=i, left=j, height=h, width=w, size=self.out_size,
        #                                   interpolation=transforms.InterpolationMode.BILINEAR)
        # annotation = transforms.F.resized_crop(annotation, top=i, left=j, height=h, width=w, size=self.out_size,
        #                                        interpolation=transforms.InterpolationMode.NEAREST)

        # return image, annotation, image_name
    
        return dict(jpg=(image - 0.5) / 0.5, 
                    txt="",
                    hint=(image - 0.5) / 0.5,
                    low=(low - 0.5) / 0.5,
                    image_name=image_name,
                    annotation=annotation)

    def __len__(self) -> int:
        return len(self.file_list)
    
    def low_fea(self, image):
        img0, img1, img2 = cv2.split(image)
        img0 = self.filter(img0)
        img1 = self.filter(img1)
        img2 = self.filter(img2)
        return np.stack((img0, img1, img2), axis=2)
    
    def get_gaussian(self, img):
        H, W = img.shape
        min_size = np.min([H, W]) * 0.05

        gaussian = np.zeros_like(img)
        tm1 = np.arange(H)
        tm1[tm1 > np.round(H / 2)] -= H
        tm1 = np.expand_dims(tm1, axis=1).repeat(W, axis=1)
        tm2 = np.arange(W)
        tm2[tm2 > np.round(W / 2)] -= W
        tm2 = np.expand_dims(tm2, axis=0).repeat(H, axis=0)

        elem_d = tm1 * tm1 + tm2 * tm2
        gaussian = np.exp(elem_d / (2 * min_size ** 2)) / 2 / np.pi / (2 * min_size ** 2)
        gaussian = gaussian / np.max(gaussian)

        return gaussian

    def filter(self, img):
        # 傅里叶变换
        img_fft = np.fft.fft2(img)
        img_fft_shift = np.fft.fftshift(img_fft)
        # 频域滤波
        img_fft_shift_process = img_fft_shift[:]
        mask = self.get_gaussian(img)
        img_fft_shift_process = img_fft_shift_process * mask
        # 傅里叶逆变换
        img_fft_ishift = np.fft.ifftshift(img_fft_shift_process)
        img_ifft = np.fft.ifft2(img_fft_ishift)
        return np.abs(img_ifft)
    
    

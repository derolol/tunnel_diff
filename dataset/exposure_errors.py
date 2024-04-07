import os
from typing import Sequence, Dict, Union
import math
import time
import random
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch.utils.data as data

from .tool.common import center_crop_arr, augment, random_crop_arr
from .tool.degradation import random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression

class ExposureErrorsDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        hq_format: str,
        lq_format: str,
        ev: Sequence[str],
        gray_style: bool,
        brightness_enhance: bool,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        dataset_type: str = "train"
    ) -> "ExposureErrorsDataset":
        
        super().__init__()
        
        self.file_list = file_list
        self.img_pair_paths = []
        self.load_file_list()

        self.brightness_enhance = brightness_enhance
        self.gray_style = gray_style
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip

        self.hq_format = hq_format
        self.lq_format = lq_format
        self.ev = ev
        
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

        self.dataset_type = dataset_type
    
    def load_file_list(self) -> List[str]:
        with open(self.file_list, "r") as fin:
            for line in fin:
                path = line.strip()
                img_name, hq_path, lq_path = path.split('\t')
                self.img_pair_paths.append({
                    'image_name': img_name,
                    'hq': os.path.join(hq_path, img_name),
                    'lq': os.path.join(lq_path, img_name),})

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_pair_path = self.img_pair_paths[index]
        if self.dataset_type=="train":
            ev_rand = random.choice(self.e)
            ev_rand = f"_{ev_rand}"
        else:
            ev_rand = ""
        hq_path = f'{img_pair_path["hq"]}.{self.hq_format}'
        lq_path = f'{img_pair_path["lq"]}{ev_rand}.{self.lq_format}'
        
        hq_pil = Image.open(hq_path).convert("RGB")
        lq_pil = Image.open(lq_path).convert("RGB")
        
        # brightness enhance
        if self.dataset_type=="train" and self.brightness_enhance:
            brightness = random.random() * 0.4 + 0.8
            lq_pil_enhance = ImageEnhance.Brightness(lq_pil)
            lq_pil = lq_pil_enhance.enhance(brightness)
        
        # crop
        if self.crop_type == "center":
            hq_pil = center_crop_arr(hq_pil, self.out_size)
            lq_pil = center_crop_arr(lq_pil, self.out_size)
        elif self.crop_type == "random":
            hq_pil, lq_pil = self.random_crop_arr([hq_pil, lq_pil], self.out_size)
        else:
            hq_pil = np.array(hq_pil)
            lq_pil = np.array(lq_pil)
            assert hq_pil.shape[:2] == (self.out_size, self.out_size)
            assert lq_pil.shape[:2] == (self.out_size, self.out_size)
        
        # brightness enhance
        if self.dataset_type=="train":
            brightness = np.random.random() * 1.2 + 0.4
            kernel_size = np.random.randint(low=16, high=256)
            x = np.random.randint(low=0, high=hq_pil.shape[0] - kernel_size - 1)
            y = np.random.randint(low=0, high=hq_pil.shape[1] - kernel_size - 1)
            brightness_map = lq_pil[x : x + kernel_size, y : y + kernel_size, :]
            brightness_map = np.power(brightness_map / float(np.max(brightness_map)), brightness)
            lq_pil[x : x + kernel_size, y : y + kernel_size, :] = np.uint8(brightness_map * 255)

            lq_pil = np.clip(a=lq_pil, a_max=255, a_min=0)

        # RGB, [0, 1]
        hq_pil = (hq_pil / 255.0).astype(np.float32)
        lq_pil = (lq_pil / 255.0).astype(np.float32)
        
        # noise
        if self.dataset_type=="train" and self.noise_range is not None:
            lq_pil = random_add_gaussian_noise(lq_pil, self.noise_range)
        
        # jpeg compression
        if self.dataset_type=="train" and self.jpeg_range is not None:
            lq_pil = random_add_jpg_compression(lq_pil, self.jpeg_range)

        # random horizontal flip
        if self.dataset_type=="train" and self.use_hflip:
            flip = random.choice([True, False])
            if flip:
                hq_pil = hq_pil.transpose(Image.FLIP_LEFT_RIGHT)
                lq_pil = lq_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # filter
        low_pil = self.low_fea(lq_pil)

        # RGB, [-1, 1]
        hq_norm = hq_pil * 2 - 1
        lq_norm = lq_pil * 2 - 1
        low_norm = low_pil * 2 - 1

        hq_c = hq_norm.transpose((2, 0, 1))
        lq_c = lq_norm.transpose((2, 0, 1))
        low_c = low_norm.transpose((2, 0, 1))

        return dict(jpg=hq_c.astype(np.float32), 
                    txt="",
                    hint=lq_c.astype(np.float32),
                    low=low_c.astype(np.float32),
                    image_name=img_pair_path["image_name"])

    def __len__(self) -> int:
        return len(self.img_pair_paths)
    
    # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
    def random_crop_arr(self, pil_images, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
        min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
        max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
        smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        new_pil_images = []

        for pil_image in pil_images:
            while min(*pil_image.size) >= 2 * smaller_dim_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = smaller_dim_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )
            new_pil_images.append(pil_image)
        
        arr = np.array(new_pil_images[0])
        crop_y = random.randrange(arr.shape[0] - image_size + 1)
        crop_x = random.randrange(arr.shape[1] - image_size + 1)

        new_pil_images = [np.array(pil_image)[crop_y : crop_y + image_size, crop_x : crop_x + image_size] for pil_image in new_pil_images]

        return new_pil_images

    def get_gauss_kernel(self, kernel_size=3, sigma=1, k=1):
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1 / (2 * np.pi * sigma**2) * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        return gauss
    
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
        mask = self(img)
        img_fft_shift_process = img_fft_shift_process * mask
        # 傅里叶逆变换
        img_fft_ishift = np.fft.ifftshift(img_fft_shift_process)
        img_ifft = np.fft.ifft2(img_fft_ishift)
        return np.abs(img_ifft)
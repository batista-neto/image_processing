from torchvision import transforms
import numpy as np
import random
import torch
from PIL import Image, ImageFilter

def resize(img, mask, base_size, scale_range):
    short_size = int(base_size * random.uniform(*scale_range))
    w, h = img.size
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def crop(img, mask, size):
    w, h = img.size
    th, tw = size if isinstance(size, tuple) else (size, size)

    if w == tw and h == th:
        return img, mask

    if w < tw or h < th:
        pad_w = max(tw - w, 0)
        pad_h = max(th - h, 0)
        img = transforms.functional.pad(img, (0, 0, pad_w, pad_h), fill=0)
        mask = transforms.functional.pad(mask, (0, 0, pad_w, pad_h), fill=255)

    w, h = img.size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    img = img.crop((x1, y1, x1 + tw, y1 + th))
    mask = mask.crop((x1, y1, x1 + tw, y1 + th))
    return img, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def blur(img, p=0.5):
    if random.random() < p:
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

def cutout(img, mask, p=0.5, size=64):
    if random.random() < p:
        w, h = img.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img.paste((0, 0, 0), (x, y, x + size, y + size))
        mask.paste(255, (x, y, x + size, y + size))  # 255 as ignore label
    return img, mask

def normalize(img, mask):
    img = transforms.ToTensor()(img)

    if isinstance(mask, Image.Image):
        mask = np.array(mask).astype(np.uint8)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).long()

    mask[mask >= 150] = 255  # importante!

    return img, mask

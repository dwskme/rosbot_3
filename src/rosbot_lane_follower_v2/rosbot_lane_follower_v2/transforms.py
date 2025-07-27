#!/usr/bin/env python
import cv2
import numpy as np
from torchvision import transforms

def apply_filters(image):
    """
    Apply the exact Gaussian blur (5×5 kernel, σ=1.0) you used at training time.
    """
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

def get_transforms(training=True):
    """
    Return a torchvision Compose that matches your train‑time normalization.
    """
    if not training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),        # if you resized
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # your existing training‑time augmentations...
        return transforms.Compose([
            # e.g. ColorJitter, RandomCrop, etc.
        ])

import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.GaussNoise(), 
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RGBShift(),
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.8),
    ToTensorV2(),
])
test_transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    ToTensorV2(),
])
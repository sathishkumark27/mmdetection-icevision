import albumentations as albu
import cv2


def get_training_augmentation(min_area=0., min_visibility=0.):
    train_transform = [
        albu.OneOf([
            albu.ISONoise(p=.5),
            albu.GaussNoise(p=0.4),
            albu.Blur(blur_limit=3, p=0.1),
        ]),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
        ], p=0.2),
        albu.OneOf([
            albu.RandomSnow(snow_point_lower=0., snow_point_upper=0.2, brightness_coeff=2., p=0.5),
            albu.RandomSunFlare(p=0.5),
        ]),
        albu.OneOf([
            albu.RGBShift(p=0.1),
            albu.ChannelShuffle(p=0.2),
        ])
    ]
    return albu.Compose(train_transform,
                        bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


def get_validation_augmentation(min_area=0., min_visibility=0.):
    test_transform = [
        albu.PadIfNeeded(1024, 1024)
    ]
    return albu.Compose(test_transform,
                        bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})

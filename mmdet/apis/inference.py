import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector
import csv
import os
from os import path


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
            #print("ignoring coco classes")
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)

def convert_to_valid_labels_old(label_text):
    #"categories": [{"id": 1, "name": "2.1", "supercategory": "2"}, {"id": 2, "name": "2.4", "supercategory": "2"}, {"id": 3, "name": "3.1", "supercategory": "3"}, 
    # {"id": 4, "name": "3.24", "supercategory": "3"}, {"id": 5, "name": "3.27", "supercategory": "3"}, {"id": 6, "name": "4.1.1", "supercategory": "4.1"}, 
    # {"id": 7, "name": "4.1.2", "supercategory": "4.1"}, {"id": 8, "name": "4.1.3", "supercategory": "4.1"}, {"id": 9, "name": "4.1.4", "supercategory": "4.1"},
    #  {"id": 10, "name": "4.1.5", "supercategory": "4.1"}, {"id": 11, "name": "4.1.6", "supercategory": "4.1"}, {"id": 12, "name": "4.2.1", "supercategory": "4.2"},
    #  {"id": 13, "name": "4.2.2", "supercategory": "4.2"}, {"id": 14, "name": "4.2.3", "supercategory": "4.2"}, {"id": 15, "name": "5.19.1", "supercategory": "5.19"}, 
    # {"id": 16, "name": "5.19.2", "supercategory": "5.19"}, {"id": 17, "name": "5.20", "supercategory": "5.20"}, {"id": 18, "name": "8.22.1", "supercategory": "8.22"}, 
    # {"id": 19, "name": "8.22.2", "supercategory": "8.22"}, {"id": 20, "name": "8.22.3", "supercategory": "8.22"}]

    #label_text can be any of the "names" from categories, Need to convert to valid classes 
    # class: traffic sign code. Valid values are: 2.1, 2.4, 3.1, 3.24, 3.27, 4.1, 4.2, 5.19, 5.20, 8.22
    if(label_text ==  "4.1.1" or  label_text ==  "4.1.2" or label_text ==  "4.1.3" or label_text ==  "4.1.4" or label_text ==  "4.1.5" or 
        label_text ==  "4.1.6" or label_text ==  "4.1.7"):
        return "4.1"
    if (label_text ==  "4.2.1" or label_text ==  "4.2.2" or label_text ==  "4.2.3"):
        return "4.2"
    if (label_text ==  "5.19.1" or label_text ==  "5.19.2"):
        return "5.19"
    if (label_text ==  "8.22.1" or label_text ==  "8.22.2" or label_text ==  "8.22.3"):
        return "8.22"
    return label_text

def convert_to_valid_labels_old(label_text):
    #label_text can be any of the "names" from categories, Need to convert to valid classes 
    # class: traffic sign code. Valid values are: 2.1, 2.4, 3.1, 3.24, 3.27, 4.1, 4.2, 5.19, 5.20, 8.22
    if(label_text ==  "4.1.1" or  label_text ==  "4.1.2" or label_text ==  "4.1.3" or label_text ==  "4.1.4" or label_text ==  "4.1.5" or 
        label_text ==  "4.1.6" or label_text ==  "4.1.7"):
        return "4.1"
    if (label_text ==  "4.2.1" or label_text ==  "4.2.2" or label_text ==  "4.2.3"):
        return "4.2"
    if (label_text ==  "5.19.1" or label_text ==  "5.19.2"):
        return "5.19"
    if (label_text ==  "8.22.1" or label_text ==  "8.22.2" or label_text ==  "8.22.3"):
        return "8.22"
    return label_text
#CLASSES = ("2.1", "2.4", "3.1", "3.24", "3.27", "4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6", "4.2.1", "4.2.2", "4.2.3", "5.19.1", "5.19.2", "5.20", "8.22.1", "8.22.2", "8.22.3")
def convert_to_valid_labels(label_text):
    # categories =  [{"id": 1, "name": "1.11.1", "supercategory": "1.11"}, {"id": 2, "name": "1.11.2", "supercategory": "1.11"}, {"id": 3, "name": "1.12.1", "supercategory": "1.12"}, {"id": 4, "name": "1.12.2", "supercategory": "1.12"}, {"id": 5, "name": "1.13", "supercategory": "1.13"}, {"id": 6, "name": "1.15", "supercategory": "1.15"}, {"id": 7, "name": "1.16", "supercategory": "1"}, {"id": 8, "name": "1.17", "supercategory": "1"}, {"id": 9, "name": "1.20.1", "supercategory": "1.20"}, {"id": 10, "name": "1.20.2", "supercategory": "1.20"}, {"id": 11, "name": "1.20.3", "supercategory": "1.20"}, {"id": 12, "name": "1.22", "supercategory": "1.22"}, {"id": 13, "name": "1.23", "supercategory": "1"}, {"id": 14, "name": "1.25", "supercategory": "1"}, {"id": 15, "name": "1.31", "supercategory": "1"}, {"id": 16, "name": "1.33", "supercategory": "1"}, {"id": 17, "name": "1.34.1", "supercategory": "1.34"}, {"id": 18, "name": "1.34.2", "supercategory": "1.34"}, {"id": 19, "name": "1.34.3", "supercategory": "1.34"}, {"id": 20, "name": "1.8", "supercategory": "1"}, {"id": 21, "name": "2.1", "supercategory": "2"}, {"id": 22, "name": "2.2", "supercategory": "2"}, {"id": 23, "name": "2.3.1", "supercategory": "2.3"}, {"id": 24, "name": "2.3.2", "supercategory": "2.3"}, {"id": 25, "name": "2.4", "supercategory": "2.4"}, {"id": 26, "name": "2.5", "supercategory": "2"}, {"id": 27, "name": "3.1", "supercategory": "3"}, {"id": 28, "name": "3.10", "supercategory": "3"}, {"id": 29, "name": "3.11", "supercategory": "3"}, {"id": 30, "name": "3.13", "supercategory": "3"}, {"id": 31, "name": "3.18.1", "supercategory": "3.18"}, {"id": 32, "name": "3.18.2", "supercategory": "3.18"}, {"id": 33, "name": "3.19", "supercategory": "3"}, {"id": 34, "name": "3.2", "supercategory": "3"}, {"id": 35, "name": "3.20", "supercategory": "3"}, {"id": 36, "name": "3.24", "supercategory": "3"}, {"id": 37, "name": "3.25", "supercategory": "3"}, {"id": 38, "name": "3.27", "supercategory": "3"}, {"id": 39, "name": "3.28", "supercategory": "3"}, {"id": 40, "name": "3.3", "supercategory": "3"}, {"id": 41, "name": "3.31", "supercategory": "3"}, {"id": 42, "name": "3.32", "supercategory": "3"}, {"id": 43, "name": "3.4", "supercategory": "3"}, {"id": 44, "name": "3.5", "supercategory": "3"}, {"id": 45, "name": "4.1.1", "supercategory": "4.1"}, {"id": 46, "name": "4.1.2", "supercategory": "4.1"}, {"id": 47, "name": "4.1.3", "supercategory": "4.1"}, {"id": 48, "name": "4.1.4", "supercategory": "4.1"}, {"id": 49, "name": "4.1.5", "supercategory": "4.1"}, {"id": 50, "name": "4.1.6", "supercategory": "4.1"}, {"id": 51, "name": "4.2.1", "supercategory": "4.2"}, {"id": 52, "name": "4.2.2", "supercategory": "4.2"}, {"id": 53, "name": "4.2.3", "supercategory": "4.2"}, {"id": 54, "name": "4.3", "supercategory": "4"}, {"id": 55, "name": "4.4.1", "supercategory": "4.4"}, {"id": 56, "name": "4.4.2", "supercategory": "4.4"}, {"id": 57, "name": "4.5.1", "supercategory": "4.5"}, {"id": 58, "name": "4.5.2", "supercategory": "4.5"}, {"id": 59, "name": "5.14", "supercategory": "5"}, {"id": 60, "name": "5.15.1", "supercategory": "5.15"}, {"id": 61, "name": "5.15.2", "supercategory": "5.15"}, {"id": 62, "name": "5.15.3", "supercategory": "5.15"}, {"id": 63, "name": "5.15.4", "supercategory": "5.15"}, {"id": 64, "name": "5.15.5", "supercategory": "5.15"}, {"id": 65, "name": "5.15.6", "supercategory": "5.15"}, {"id": 66, "name": "5.15.7", "supercategory": "5.15"}, {"id": 67, "name": "5.16", "supercategory": "5"}, {"id": 68, "name": "5.19.1", "supercategory": "5.19"}, {"id": 69, "name": "5.19.2", "supercategory": "5.19"}, {"id": 70, "name": "5.20", "supercategory": "5"}, {"id": 71, "name": "5.21", "supercategory": "5"}, {"id": 72, "name": "5.23.1", "supercategory": "5.23"}, {"id": 73, "name": "5.24.1", "supercategory": "5.24"}, {"id": 74, "name": "5.3", "supercategory": "5"}, {"id": 75, "name": "5.31", "supercategory": "5"}, {"id": 76, "name": "5.32", "supercategory": "5"}, {"id": 77, "name": "5.4", "supercategory": "5"}, {"id": 78, "name": "5.5", "supercategory": "5"}, {"id": 79, "name": "5.6", "supercategory": "5"}, {"id": 80, "name": "5.7.1", "supercategory": "5.7"}, {"id": 81, "name": "5.7.2", "supercategory": "5.7"}, {"id": 82, "name": "6.10.1", "supercategory": "6.10"}, {"id": 83, "name": "6.10.2", "supercategory": "6.10"}, {"id": 84, "name": "6.11", "supercategory": "6"}, {"id": 85, "name": "6.13", "supercategory": "6"}, {"id": 86, "name": "6.16", "supercategory": "6"}, {"id": 87, "name": "6.3.1", "supercategory": "6.3"}, {"id": 88, "name": "6.4", "supercategory": "6"}, {"id": 89, "name": "6.6", "supercategory": "6"}, {"id": 90, "name": "6.7", "supercategory": "6"}, {"id": 91, "name": "6.8.1", "supercategory": "6.8"}, {"id": 92, "name": "6.9.1", "supercategory": "6.9"}, {"id": 93, "name": "6.9.2", "supercategory": "6.9"}, {"id": 94, "name": "7.19", "supercategory": "7"}, {"id": 95, "name": "7.2", "supercategory": "7"}, {"id": 96, "name": "7.3", "supercategory": "7"}, {"id": 97, "name": "7.5", "supercategory": "7"}, {"id": 98, "name": "8", "supercategory": "8"}, {"id": 99, "name": "8.1.1", "supercategory": "8.1"}, {"id": 100, "name": "8.11", "supercategory": "8"}, {"id": 101, "name": "8.13", "supercategory": "8"}, {"id": 102, "name": "8.14", "supercategory": "8"}, {"id": 103, "name": "8.17", "supercategory": "8"}, {"id": 104, "name": "8.2.1", "supercategory": "8.2"}, {"id": 105, "name": "8.2.2", "supercategory": "8.2"}, {"id": 106, "name": "8.2.3", "supercategory": "8.2"}, {"id": 107, "name": "8.2.4", "supercategory": "8.2"}, {"id": 108, "name": "8.2.5", "supercategory": "8.2"}, {"id": 109, "name": "8.2.6", "supercategory": "8.2"}, {"id": 110, "name": "8.21.1", "supercategory": "8.21"}, {"id": 111, "name": "8.22.1", "supercategory": "8.22"}, {"id": 112, "name": "8.22.2", "supercategory": "8.22"}, {"id": 113, "name": "8.22.3", "supercategory": "8.22"}, {"id": 114, "name": "8.23", "supercategory": "8"}, {"id": 115, "name": "8.24", "supercategory": "8"}, {"id": 116, "name": "8.3.1", "supercategory": "8.3"}, {"id": 117, "name": "8.3.2", "supercategory": "8.3"}, {"id": 118, "name": "8.4.1", "supercategory": "8.4"}, {"id": 119, "name": "8.4.3", "supercategory": "8.4"}, {"id": 120, "name": "8.5.2", "supercategory": "8.5"}, {"id": 121, "name": "8.5.4", "supercategory": "8.5"}, {"id": 122, "name": "8.6.1", "supercategory": "8.6"}, {"id": 123, "name": "8.6.5", "supercategory": "8.6"}, {"id": 124, "name": "8.7", "supercategory": "8"}, {"id": 125, "name": "8.8", "supercategory": "8"}]

    categories =  [{'id': 1, 'name': '1.11.1', 'supercategory': '1.11'}, {'id': 2, 'name': '1.11.2', 'supercategory': '1.11'}, {'id': 3, 'name': '1.12.1', 'supercategory': '1.12'}, {'id': 4, 'name': '1.12.2', 'supercategory': '1.12'}, {'id': 5, 'name': '1.13', 'supercategory': '1.13'}, {'id': 6, 'name': '1.15', 'supercategory': '1.15'}, {'id': 7, 'name': '1.16', 'supercategory': '1.16'}, {'id': 8, 'name': '1.17', 'supercategory': '1.17'}, {'id': 9, 'name': '1.20.1', 'supercategory': '1.20'}, {'id': 10, 'name': '1.20.2', 'supercategory': '1.20'}, {'id': 11, 'name': '1.20.3', 'supercategory': '1.20'}, {'id': 12, 'name': '1.22', 'supercategory': '1.22'}, {'id': 13, 'name': '1.23', 'supercategory': '1.23'}, {'id': 14, 'name': '1.25', 'supercategory': '1.25'}, {'id': 15, 'name': '1.31', 'supercategory': '1.31'}, {'id': 16, 'name': '1.33', 'supercategory': '1.33'}, {'id': 17, 'name': '1.34.1', 'supercategory': '1.34'}, {'id': 18, 'name': '1.34.2', 'supercategory': '1.34'}, {'id': 19, 'name': '1.34.3', 'supercategory': '1.34'}, {'id': 20, 'name': '1.8', 'supercategory': '1.8'}, {'id': 21, 'name': '2.1', 'supercategory': '2.1'}, {'id': 22, 'name': '2.2', 'supercategory': '2.2'}, {'id': 23, 'name': '2.3.1', 'supercategory': '2.3'}, {'id': 24, 'name': '2.3.2', 'supercategory': '2.3'}, {'id': 25, 'name': '2.4', 'supercategory': '2.4'}, {'id': 26, 'name': '2.5', 'supercategory': '2.5'}, {'id': 27, 'name': '3.1', 'supercategory': '3.1'}, {'id': 28, 'name': '3.10', 'supercategory': '3.10'}, {'id': 29, 'name': '3.11', 'supercategory': '3.11'}, {'id': 30, 'name': '3.13', 'supercategory': '3.13'}, {'id': 31, 'name': '3.18.1', 'supercategory': '3.18'}, {'id': 32, 'name': '3.18.2', 'supercategory': '3.18'}, {'id': 33, 'name': '3.19', 'supercategory': '3.19'}, {'id': 34, 'name': '3.2', 'supercategory': '3.2'}, {'id': 35, 'name': '3.20', 'supercategory': '3.20'}, {'id': 36, 'name': '3.24', 'supercategory': '3.24'}, {'id': 37, 'name': '3.25', 'supercategory': '3.25'}, {'id': 38, 'name': '3.27', 'supercategory': '3.27'}, {'id': 39, 'name': '3.28', 'supercategory': '3.28'}, {'id': 40, 'name': '3.3', 'supercategory': '3.3'}, {'id': 41, 'name': '3.31', 'supercategory': '3.31'}, {'id': 42, 'name': '3.32', 'supercategory': '3.32'}, {'id': 43, 'name': '3.4', 'supercategory': '3.4'}, {'id': 44, 'name': '3.5', 'supercategory': '3.5'}, {'id': 45, 'name': '4.1.1', 'supercategory': '4.1'}, {'id': 46, 'name': '4.1.2', 'supercategory': '4.1'}, {'id': 47, 'name': '4.1.3', 'supercategory': '4.1'}, {'id': 48, 'name': '4.1.4', 'supercategory': '4.1'}, {'id': 49, 'name': '4.1.5', 'supercategory': '4.1'}, {'id': 50, 'name': '4.1.6', 'supercategory': '4.1'}, {'id': 51, 'name': '4.2.1', 'supercategory': '4.2'}, {'id': 52, 'name': '4.2.2', 'supercategory': '4.2'}, {'id': 53, 'name': '4.2.3', 'supercategory': '4.2'}, {'id': 54, 'name': '4.3', 'supercategory': '4.3'}, {'id': 55, 'name': '4.4.1', 'supercategory': '4.4'}, {'id': 56, 'name': '4.4.2', 'supercategory': '4.4'}, {'id': 57, 'name': '4.5.1', 'supercategory': '4.5'}, {'id': 58, 'name': '4.5.2', 'supercategory': '4.5'}, {'id': 59, 'name': '5.14', 'supercategory': '5.14'}, {'id': 60, 'name': '5.15.1', 'supercategory': '5.15'}, {'id': 61, 'name': '5.15.2', 'supercategory': '5.15'}, {'id': 62, 'name': '5.15.3', 'supercategory': '5.15'}, {'id': 63, 'name': '5.15.4', 'supercategory': '5.15'}, {'id': 64, 'name': '5.15.5', 'supercategory': '5.15'}, {'id': 65, 'name': '5.15.6', 'supercategory': '5.15'}, {'id': 66, 'name': '5.15.7', 'supercategory': '5.15'}, {'id': 67, 'name': '5.16', 'supercategory': '5.16'}, {'id': 68, 'name': '5.19.1', 'supercategory': '5.19'}, {'id': 69, 'name': '5.19.2', 'supercategory': '5.19'}, {'id': 70, 'name': '5.20', 'supercategory': '5.20'}, {'id': 71, 'name': '5.21', 'supercategory': '5.21'}, {'id': 72, 'name': '5.23.1', 'supercategory': '5.23'}, {'id': 73, 'name': '5.24.1', 'supercategory': '5.24'}, {'id': 74, 'name': '5.3', 'supercategory': '5.3'}, {'id': 75, 'name': '5.31', 'supercategory': '5.31'}, {'id': 76, 'name': '5.32', 'supercategory': '5.32'}, {'id': 77, 'name': '5.4', 'supercategory': '5.4'}, {'id': 78, 'name': '5.5', 'supercategory': '5.5'}, {'id': 79, 'name': '5.6', 'supercategory': '5.6'}, {'id': 80, 'name': '5.7.1', 'supercategory': '5.7'}, {'id': 81, 'name': '5.7.2', 'supercategory': '5.7'}, {'id': 82, 'name': '6.10.1', 'supercategory': '6.10'}, {'id': 83, 'name': '6.10.2', 'supercategory': '6.10'}, {'id': 84, 'name': '6.11', 'supercategory': '6.11'}, {'id': 85, 'name': '6.13', 'supercategory': '6.13'}, {'id': 86, 'name': '6.16', 'supercategory': '6.16'}, {'id': 87, 'name': '6.3.1', 'supercategory': '6.3'}, {'id': 88, 'name': '6.4', 'supercategory': '6.4'}, {'id': 89, 'name': '6.6', 'supercategory': '6.6'}, {'id': 90, 'name': '6.7', 'supercategory': '6.7'}, {'id': 91, 'name': '6.8.1', 'supercategory': '6.8'}, {'id': 92, 'name': '6.9.1', 'supercategory': '6.9'}, {'id': 93, 'name': '6.9.2', 'supercategory': '6.9'}, {'id': 94, 'name': '7.19', 'supercategory': '7.19'}, {'id': 95, 'name': '7.2', 'supercategory': '7.2'}, {'id': 96, 'name': '7.3', 'supercategory': '7.3'}, {'id': 97, 'name': '7.5', 'supercategory': '7.5'}, {'id': 98, 'name': '8', 'supercategory': '8'}, {'id': 99, 'name': '8.1.1', 'supercategory': '8.1'}, {'id': 100, 'name': '8.11', 'supercategory': '8.11'}, {'id': 101, 'name': '8.13', 'supercategory': '8.13'}, {'id': 102, 'name': '8.14', 'supercategory': '8.14'}, {'id': 103, 'name': '8.17', 'supercategory': '8.17'}, {'id': 104, 'name': '8.2.1', 'supercategory': '8.2'}, {'id': 105, 'name': '8.2.2', 'supercategory': '8.2'}, {'id': 106, 'name': '8.2.3', 'supercategory': '8.2'}, {'id': 107, 'name': '8.2.4', 'supercategory': '8.2'}, {'id': 108, 'name': '8.2.5', 'supercategory': '8.2'}, {'id': 109, 'name': '8.2.6', 'supercategory': '8.2'}, {'id': 110, 'name': '8.21.1', 'supercategory': '8.21'}, {'id': 111, 'name': '8.22.1', 'supercategory': '8.22'}, {'id': 112, 'name': '8.22.2', 'supercategory': '8.22'}, {'id': 113, 'name': '8.22.3', 'supercategory': '8.22'}, {'id': 114, 'name': '8.23', 'supercategory': '8.23'}, {'id': 115, 'name': '8.24', 'supercategory': '8.24'}, {'id': 116, 'name': '8.3.1', 'supercategory': '8.3'}, {'id': 117, 'name': '8.3.2', 'supercategory': '8.3'}, {'id': 118, 'name': '8.4.1', 'supercategory': '8.4'}, {'id': 119, 'name': '8.4.3', 'supercategory': '8.4'}, {'id': 120, 'name': '8.5.2', 'supercategory': '8.5'}, {'id': 121, 'name': '8.5.4', 'supercategory': '8.5'}, {'id': 122, 'name': '8.6.1', 'supercategory': '8.6'}, {'id': 123, 'name': '8.6.5', 'supercategory': '8.6'}, {'id': 124, 'name': '8.7', 'supercategory': '8.7'}, {'id': 125, 'name': '8.8', 'supercategory': '8.8'}]

    for i in range(len(categories)):        
        if label_text == categories[i]["name"]:
            label_text = categories[i]["supercategory"]
    return label_text




def write_result(img, result, class_names, score_thr=0.3, out_file="result.tsv"):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    #img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    # if segm_result is not None:
    #     segms = mmcv.concat_list(segm_result)
    #     inds = np.where(bboxes[:, -1] > score_thr)[0]
    #     for i in inds:
    #         color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    #         mask = maskUtils.decode(segms[i]).astype(np.bool)
    #         img[mask] = img[mask] * 0.5 + color_mask * 0.5

    # Get bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    
    labels = np.concatenate(labels)

    #MMCV
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    #CLASSES = ("2.1", "2.4", "3.1", "3.24", "3.27", "4.1.1", "4.1.2", "4.1.3", "4.1.4", "4.1.5", "4.1.6", "4.2.1", "4.2.2", "4.2.3", "5.19.1", "5.19.2", "5.20", "8.22.1", "8.22.2", "8.22.3")
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        fileEmpty = True
        if path.exists(out_file) :
            fileEmpty = os.stat(out_file).st_size == 0
        with open(out_file, mode = 'a') as result_file:
                result_file_writer = csv.writer(result_file, delimiter = '\t')
                if fileEmpty : 
                    result_file_writer.writerow(['frame', 'xtl', 'ytl', 'xbr', 'ybr', 'class', 'temporary', 'data'])
                for bbox, label in zip(bboxes, labels):
                    bbox_int = bbox.astype(np.int32)
                    xtl = bbox_int[0]
                    ytl = bbox_int[1]
                    xbr = bbox_int[2]
                    ybr = bbox_int[3]
                    temporary = ''
                    data = ''
                    label_text = class_names[label] if class_names is not None else 'cls {}'.format(label)
                    #if label_text not in CLASSES:
                        #continue
                    #print("label = %d labe_text = %s" %(label,label_text))
                    #label_text = convert_to_valid_labels(label_text)
                    result_file_writer.writerow([img, xtl, ytl, xbr, ybr, label_text, temporary, data])
            
                


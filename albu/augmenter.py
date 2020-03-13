import fnmatch
import json
import os
import re
import pandas as pd
import cv2
import shutil
from .augmentations import get_training_augmentation, get_validation_augmentation
from converter import create_annotation_info

CATEGORIES = [
    {
        "id": 1,
        "name": "1.1",
        "supercategory": "1"
    },
    {
        "id": 2,
        "name": "1.11.1",
        "supercategory": "1.11"
    },
    {
        "id": 3,
        "name": "1.11.2",
        "supercategory": "1.11"
    },
    {
        "id": 4,
        "name": "1.12.1",
        "supercategory": "1.12"
    },
    {
        "id": 5,
        "name": "1.12.2",
        "supercategory": "1.12"
    },
    {
        "id": 6,
        "name": "1.13",
        "supercategory": "1"
    },
    {
        "id": 7,
        "name": "1.15",
        "supercategory": "1"
    },
    {
        "id": 8,
        "name": "1.16",
        "supercategory": "1"
    },
    {
        "id": 9,
        "name": "1.17",
        "supercategory": "1"
    },
    {
        "id": 10,
        "name": "1.20.1",
        "supercategory": "1.20"
    },
    {
        "id": 11,
        "name": "1.20.2",
        "supercategory": "1.20"
    },
    {
        "id": 12,
        "name": "1.20.3",
        "supercategory": "1.20"
    },
    {
        "id": 13,
        "name": "1.22",
        "supercategory": "1"
    },
    {
        "id": 14,
        "name": "1.23",
        "supercategory": "1"
    },
    {
        "id": 15,
        "name": "1.25",
        "supercategory": "1"
    },
    {
        "id": 16,
        "name": "1.3.1",
        "supercategory": "1.3"
    },
    {
        "id": 17,
        "name": "1.31",
        "supercategory": "1"
    },
    {
        "id": 18,
        "name": "1.33",
        "supercategory": "1"
    },
    {
        "id": 19,
        "name": "1.34.1",
        "supercategory": "1.34"
    },
    {
        "id": 20,
        "name": "1.34.2",
        "supercategory": "1.34"
    },
    {
        "id": 21,
        "name": "1.34.3",
        "supercategory": "1.34"
    },
    {
        "id": 22,
        "name": "1.8",
        "supercategory": "1"
    },
    {
        "id": 23,
        "name": "2.1",
        "supercategory": "2"
    },
    {
        "id": 24,
        "name": "2.2",
        "supercategory": "2"
    },
    {
        "id": 25,
        "name": "2.3.1",
        "supercategory": "2.3"
    },
    {
        "id": 26,
        "name": "2.3.2",
        "supercategory": "2.3"
    },
    {
        "id": 27,
        "name": "2.4",
        "supercategory": "2"
    },
    {
        "id": 28,
        "name": "2.5",
        "supercategory": "2"
    },
    {
        "id": 29,
        "name": "2.6",
        "supercategory": "2"
    },
    {
        "id": 30,
        "name": "3.1",
        "supercategory": "3"
    },
    {
        "id": 31,
        "name": "3.10",
        "supercategory": "3"
    },
    {
        "id": 32,
        "name": "3.11",
        "supercategory": "3"
    },
    {
        "id": 33,
        "name": "3.13",
        "supercategory": "3"
    },
    {
        "id": 34,
        "name": "3.18.1",
        "supercategory": "3.18"
    },
    {
        "id": 35,
        "name": "3.18.2",
        "supercategory": "3.18"
    },
    {
        "id": 36,
        "name": "3.19",
        "supercategory": "3"
    },
    {
        "id": 37,
        "name": "3.2",
        "supercategory": "3"
    },
    {
        "id": 38,
        "name": "3.20",
        "supercategory": "3"
    },
    {
        "id": 39,
        "name": "3.24",
        "supercategory": "3"
    },
    {
        "id": 40,
        "name": "3.25",
        "supercategory": "3"
    },
    {
        "id": 41,
        "name": "3.27",
        "supercategory": "3"
    },
    {
        "id": 42,
        "name": "3.28",
        "supercategory": "3"
    },
    {
        "id": 43,
        "name": "3.3",
        "supercategory": "3"
    },
    {
        "id": 44,
        "name": "3.31",
        "supercategory": "3"
    },
    {
        "id": 45,
        "name": "3.32",
        "supercategory": "3"
    },
    {
        "id": 46,
        "name": "3.4",
        "supercategory": "3"
    },
    {
        "id": 47,
        "name": "3.5",
        "supercategory": "3"
    },
    {
        "id": 48,
        "name": "3.9",
        "supercategory": "3"
    },
    {
        "id": 49,
        "name": "4.1.1",
        "supercategory": "4.1"
    },
    {
        "id": 50,
        "name": "4.1.2",
        "supercategory": "4.1"
    },
    {
        "id": 51,
        "name": "4.1.3",
        "supercategory": "4.1"
    },
    {
        "id": 52,
        "name": "4.1.4",
        "supercategory": "4.1"
    },
    {
        "id": 53,
        "name": "4.1.5",
        "supercategory": "4.1"
    },
    {
        "id": 54,
        "name": "4.1.6",
        "supercategory": "4.1"
    },
    {
        "id": 55,
        "name": "4.2.1",
        "supercategory": "4.2"
    },
    {
        "id": 56,
        "name": "4.2.2",
        "supercategory": "4.2"
    },
    {
        "id": 57,
        "name": "4.2.3",
        "supercategory": "4.2"
    },
    {
        "id": 58,
        "name": "4.3",
        "supercategory": "4"
    },
    {
        "id": 59,
        "name": "4.4.1",
        "supercategory": "4.4"
    },
    {
        "id": 60,
        "name": "4.4.2",
        "supercategory": "4.4"
    },
    {
        "id": 61,
        "name": "4.5.1",
        "supercategory": "4.5"
    },
    {
        "id": 62,
        "name": "4.5.2",
        "supercategory": "4.5"
    },
    {
        "id": 63,
        "name": "5.14",
        "supercategory": "5"
    },
    {
        "id": 64,
        "name": "5.15.1",
        "supercategory": "5.15"
    },
    {
        "id": 65,
        "name": "5.15.2",
        "supercategory": "5.15"
    },
    {
        "id": 66,
        "name": "5.15.3",
        "supercategory": "5.15"
    },
    {
        "id": 67,
        "name": "5.15.4",
        "supercategory": "5.15"
    },
    {
        "id": 68,
        "name": "5.15.5",
        "supercategory": "5.15"
    },
    {
        "id": 69,
        "name": "5.15.6",
        "supercategory": "5.15"
    },
    {
        "id": 70,
        "name": "5.15.7",
        "supercategory": "5.15"
    },
    {
        "id": 71,
        "name": "5.16",
        "supercategory": "5"
    },
    {
        "id": 72,
        "name": "5.19.1",
        "supercategory": "5.19"
    },
    {
        "id": 73,
        "name": "5.19.2",
        "supercategory": "5.19"
    },
    {
        "id": 74,
        "name": "5.20",
        "supercategory": "5"
    },
    {
        "id": 75,
        "name": "5.21",
        "supercategory": "5"
    },
    {
        "id": 76,
        "name": "5.23.1",
        "supercategory": "5.23"
    },
    {
        "id": 77,
        "name": "5.24.1",
        "supercategory": "5.24"
    },
    {
        "id": 78,
        "name": "5.3",
        "supercategory": "5"
    },
    {
        "id": 79,
        "name": "5.31",
        "supercategory": "5"
    },
    {
        "id": 80,
        "name": "5.32",
        "supercategory": "5"
    },
    {
        "id": 81,
        "name": "5.4",
        "supercategory": "5"
    },
    {
        "id": 82,
        "name": "5.5",
        "supercategory": "5"
    },
    {
        "id": 83,
        "name": "5.6",
        "supercategory": "5"
    },
    {
        "id": 84,
        "name": "5.7.1",
        "supercategory": "5.7"
    },
    {
        "id": 85,
        "name": "5.7.2",
        "supercategory": "5.7"
    },
    {
        "id": 86,
        "name": "6.10.1",
        "supercategory": "6.10"
    },
    {
        "id": 87,
        "name": "6.10.2",
        "supercategory": "6.10"
    },
    {
        "id": 88,
        "name": "6.11",
        "supercategory": "6"
    },
    {
        "id": 89,
        "name": "6.12",
        "supercategory": "6"
    },
    {
        "id": 90,
        "name": "6.13",
        "supercategory": "6"
    },
    {
        "id": 91,
        "name": "6.16",
        "supercategory": "6"
    },
    {
        "id": 92,
        "name": "6.18.3",
        "supercategory": "6.18"
    },
    {
        "id": 93,
        "name": "6.3.1",
        "supercategory": "6.3"
    },
    {
        "id": 94,
        "name": "6.4",
        "supercategory": "6"
    },
    {
        "id": 95,
        "name": "6.6",
        "supercategory": "6"
    },
    {
        "id": 96,
        "name": "6.7",
        "supercategory": "6"
    },
    {
        "id": 97,
        "name": "6.8.1",
        "supercategory": "6.8"
    },
    {
        "id": 98,
        "name": "6.9.1",
        "supercategory": "6.9"
    },
    {
        "id": 99,
        "name": "6.9.2",
        "supercategory": "6.9"
    },
    {
        "id": 100,
        "name": "7.19",
        "supercategory": "7"
    },
    {
        "id": 101,
        "name": "7.2",
        "supercategory": "7"
    },
    {
        "id": 102,
        "name": "7.3",
        "supercategory": "7"
    },
    {
        "id": 103,
        "name": "7.5",
        "supercategory": "7"
    },
    {
        "id": 104,
        "name": "7.7",
        "supercategory": "7"
    },
    {
        "id": 105,
        "name": "8",
        "supercategory": "8"
    },
    {
        "id": 106,
        "name": "8.1.1",
        "supercategory": "8.1"
    },
    {
        "id": 107,
        "name": "8.1.4",
        "supercategory": "8.1"
    },
    {
        "id": 108,
        "name": "8.11",
        "supercategory": "8"
    },
    {
        "id": 109,
        "name": "8.13",
        "supercategory": "8"
    },
    {
        "id": 110,
        "name": "8.14",
        "supercategory": "8"
    },
    {
        "id": 111,
        "name": "8.17",
        "supercategory": "8"
    },
    {
        "id": 112,
        "name": "8.2.1",
        "supercategory": "8.2"
    },
    {
        "id": 113,
        "name": "8.2.2",
        "supercategory": "8.2"
    },
    {
        "id": 114,
        "name": "8.2.3",
        "supercategory": "8.2"
    },
    {
        "id": 115,
        "name": "8.2.4",
        "supercategory": "8.2"
    },
    {
        "id": 116,
        "name": "8.2.5",
        "supercategory": "8.2"
    },
    {
        "id": 117,
        "name": "8.2.6",
        "supercategory": "8.2"
    },
    {
        "id": 118,
        "name": "8.21.1",
        "supercategory": "8.21"
    },
    {
        "id": 119,
        "name": "8.22.1",
        "supercategory": "8.22"
    },
    {
        "id": 120,
        "name": "8.22.2",
        "supercategory": "8.22"
    },
    {
        "id": 121,
        "name": "8.22.3",
        "supercategory": "8.22"
    },
    {
        "id": 122,
        "name": "8.23",
        "supercategory": "8"
    },
    {
        "id": 123,
        "name": "8.24",
        "supercategory": "8"
    },
    {
        "id": 124,
        "name": "8.3.1",
        "supercategory": "8.3"
    },
    {
        "id": 125,
        "name": "8.3.2",
        "supercategory": "8.3"
    },
    {
        "id": 126,
        "name": "8.3.3",
        "supercategory": "8.3"
    },
    {
        "id": 127,
        "name": "8.4.1",
        "supercategory": "8.4"
    },
    {
        "id": 128,
        "name": "8.4.3",
        "supercategory": "8.4"
    },
    {
        "id": 129,
        "name": "8.5.2",
        "supercategory": "8.5"
    },
    {
        "id": 130,
        "name": "8.5.4",
        "supercategory": "8.5"
    },
    {
        "id": 131,
        "name": "8.6.1",
        "supercategory": "8.6"
    },
    {
        "id": 132,
        "name": "8.6.5",
        "supercategory": "8.6"
    },
    {
        "id": 133,
        "name": "8.7",
        "supercategory": "8"
    },
    {
        "id": 134,
        "name": "8.8",
        "supercategory": "8"
    }
]


def read_tsv(file):
    df = pd.read_csv(file, sep="\t")
    return df


def filter_for_annotations(root, annotation_dir):
    file_types = ['*.tsv']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    file_name_prefix = '.*'  # does nothing here
    files = [os.path.join(root, annotation_dir, file) for file in os.listdir(os.path.join(root, annotation_dir))]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def save_augmented(annotations, coco_output, image_save_path, annotation_save_path):

    img = annotations['image'].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_save_path, img)

# Commenting out annotations copying script.
    # tsv = list()
    # for idx, bbox in enumerate(annotations['bboxes']):
    #     x_min, y_min, w, h = bbox
    #     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    #     occluded = bool(coco_output["annotations"][idx]['iscrowd'])
    #     temporary = coco_output["temporaries"][idx]

    #     tsv.append({
    #         "class": annotations['category_id'][idx],
    #         "xtl": x_min,
    #         "ytl": y_min,
    #         "xbr": x_max,
    #         "ybr": y_max,
    #         "temporary": temporary,
    #         "occluded": occluded,
    #         "data": "",
    #     })

    # if len(tsv) > 0:
    #     df = pd.DataFrame(tsv)

    #     df.to_csv(annotation_save_path, sep="\t", index=False,
    #               columns=['class', 'xtl', 'ytl', 'xbr', 'ybr', 'temporary', 'occluded', 'data'])


def orig2aug(categories, root,
             image_dir, annotation_dir,
             image_save_dir, annotation_save_dir,
             extension=".jpg", data="train"):

    im_save_path = os.path.join(root, image_save_dir)
    ann_save_path = os.path.join(root, annotation_save_dir)

    if not os.path.exists(im_save_path):
        os.mkdir(im_save_path)
    if not os.path.exists(ann_save_path):
        os.mkdir(ann_save_path)

    if categories is None:
        categories = CATEGORIES
    else:
        with open(categories, 'r') as file:
            categories = json.load(file)

    aug = get_training_augmentation() if data == "train" \
        else get_validation_augmentation()

    annotation_files = filter_for_annotations(root, annotation_dir)

    for idx, annotation_filename in enumerate(annotation_files):
        print('%dth of %d annotation files is being processed' %(idx, len(annotation_files)))
        tsv = read_tsv(annotation_filename)

        basename = os.path.basename(annotation_filename)
        image_name = os.path.splitext(basename)[0] + extension

        image_filename = os.path.join(root, image_dir, image_name)
    

        image = cv2.imread(image_filename)
        print(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape[0:2][::-1]

        coco_output = {
            "temporaries": [],
            "annotations": []
        }

        for index, row in tsv.iterrows():
            value = [x for x in categories if x["name"] == str(row["class"])]

            if value:
                value = value[0]
                class_id = value["id"]
                name = value["name"]
                category_info = {"id": class_id, "is_crowd": 0 if row["occluded"] is None else int(row["occluded"])}
                annotation_info = create_annotation_info(0, 0, category_info,
                                                         image_size, row.iloc[1:5])  # Dict type

                if annotation_info is not None:
                    annotation_info['name'] = name  # add "name" key to athe annotation_info
                    coco_output["annotations"].append(annotation_info)
                    coco_output["temporaries"].append(row["temporary"])

        annotations = {'image': image, 'bboxes': [d['bbox'] for d in coco_output['annotations']],
                       'category_id': [d['name'] for d in coco_output['annotations']]}

        if len(annotations['bboxes']) > 0:
            # if the image has bounding box data
            augmented = aug(**annotations)

            base_name = os.path.splitext(basename)[0] + '_aug'
            image_save_path = os.path.join(root, image_save_dir, base_name + extension)
            annotation_save_path = os.path.join(root, annotation_save_dir, base_name + ".tsv")
            
            # Copy original annotation data and rename as augmented annotation
            shutil.copy(annotation_filename, annotation_save_path)


            # For each frame that it has
            save_augmented(augmented, coco_output, image_save_path, annotation_save_path)

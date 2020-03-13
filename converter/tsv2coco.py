import datetime
import json
import numpy as np
import pandas as pd
import os
import re
import fnmatch
from PIL import Image
from .pycococreatortools import create_image_info, create_annotation_info


INFO = {
    "description": "IceVision Dataset",
    "url": "https://github.com/ptJexio/icevision",
    "version": "0.0.1",
    "year": 2019,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': '2.1',
        'supercategory': '2',
    },
    {
        'id': 2,
        'name': '2.4',
        'supercategory': '2',
    },
    {
        'id': 3,
        'name': '3.1',
        'supercategory': '3',
    },
    {
        'id': 4,
        'name': '3.24',
        'supercategory': '3',
    },
    {
        'id': 5,
        'name': '3.27',
        'supercategory': '3',
    },
    {
        'id': 6,
        'name': '4.1.1',
        'supercategory': '4.1',
    },
    {
        'id': 7,
        'name': '4.1.2',
        'supercategory': '4.1',
    },
    {
        'id': 8,
        'name': '4.1.3',
        'supercategory': '4.1',
    },
    {
        'id': 9,
        'name': '4.1.4',
        'supercategory': '4.1',
    },
    {
        'id': 10,
        'name': '4.1.5',
        'supercategory': '4.1',
    },
    {
        'id': 11,
        'name': '4.1.6',
        'supercategory': '4.1',
    },
    {
        'id': 12,
        'name': '4.2.1',
        'supercategory': '4.2',
    },
    {
        'id': 13,
        'name': '4.2.2',
        'supercategory': '4.2',
    },
    {
        'id': 14,
        'name': '4.2.3',
        'supercategory': '4.2',
    },
    {
        'id': 15,
        'name': '5.19.1',
        'supercategory': '5.19',
    },
    {
        'id': 16,
        'name': '5.19.2',
        'supercategory': '5.19',
    },
    {
        'id': 17,
        'name': '5.20',
        'supercategory': '5.20',
    },
    {
        'id': 18,
        'name': '8.22.1',
        'supercategory': '8.22',
    },
    {
        'id': 19,
        'name': '8.22.2',
        'supercategory': '8.22',
    },
    {
        'id': 20,
        'name': '8.22.3',
        'supercategory': '8.22',
    },
]


def filter_for_image(root, files):
    file_types = ["*.jpeg", "*.jpg", "*.png"]
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, annotation_dir):
    file_types = ['*.tsv']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    file_name_prefix = '.*'
    files = [os.path.join(root, annotation_dir, file) for file in os.listdir(os.path.join(root, annotation_dir))]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def read_tsv(file):
    df = pd.read_csv(file, sep="\t")
    return df


def tsv2coco(categories_path, root, image_dir, annotation_dir, extension=".jpg", is_compute_norm=False):

    if categories_path is None:
        categories = CATEGORIES
    else:
        with open(categories_path, 'r') as file:
            categories = json.load(file)

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    annotation_files = filter_for_annotations(root, annotation_dir)

    global_info = {c["name"]: 0 for c in categories}
    other_classes = {}

    # compute mean, std
    list_image = list()

    for annotation_filename in annotation_files:

        tsv = read_tsv(annotation_filename)

        basename = os.path.basename(annotation_filename)
        image_name = os.path.splitext(basename)[0] + extension

        image_filename = os.path.join(root, image_dir, image_name)

        image = Image.open(image_filename)

        if is_compute_norm:
            im_array = np.asarray(image)
            im_array = np.resize(im_array, (512, 512, 3))
            list_image.append(im_array)

        image_info = create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        for index, row in tsv.iterrows():

            classes = [x for x in categories if x["name"] == str(row["class"])]

            if len(classes) == 0:
                if isinstance(row["class"], str):
                    if row["class"] in other_classes:
                        other_classes[row["class"]] += 1
                    else:
                        other_classes[row["class"]] = 1
                continue

            class_id = classes[0]["id"]
            name = classes[0]["name"]

            global_info[name] += 1

            category_info = {"id": class_id, "is_crowd": 0 if row["occluded"] is None else int(row["occluded"])}

            annotation_info = create_annotation_info(
                segmentation_id, image_id, category_info,
                image.size, row.iloc[1:5])

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    if is_compute_norm:
        images = [image.reshape(3, -1) for image in list_image]
        images = np.concatenate(images, 1)
        mean = images.mean(1)
        std = images.std(1)

        print('mean - {0}, std - {1}'.format(mean, std))

    print('-' * 48)
    print('Signs of the competition')
    print('-' * 48)
    sum = 0
    for key, item in global_info.items():
        sum += item
        print('sign - {0}, count - {1}'.format(key, item))
    print('Sum - {0}'.format(sum))

    print('-' * 48)
    print('Other signs on image')
    print('-' * 48)
    sum = 0
    for key, item in other_classes.items():
        sum += item
        print('sign - {0}, count - {1}'.format(key, item))
    print('Sum - {0}'.format(sum))

    return coco_output
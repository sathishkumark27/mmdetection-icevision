import datetime


def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_info,
                           image_size, bounding_box):

    is_crowd = category_info["is_crowd"]

    width, height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
    area = width * height

    bounding_box = list(bounding_box)
    bounding_box = [*bounding_box[0:2], bounding_box[2]-bounding_box[0], bounding_box[3]-bounding_box[1]]

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": is_crowd,
        "area": area,
        "bbox": bounding_box,
        "segmentation": None,
        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info

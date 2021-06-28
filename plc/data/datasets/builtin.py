# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib

import detectron2.data.datasets.coco
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging

# TODO: find out wheter i need this for my own dataset ! i do to register my subset
logger = logging.getLogger(__name__)

def register_own_dataset(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """


    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts

    # DatasetCatalog.register(
    #     name, lambda: load_owndataset(json_file, image_root, name)
    # )
    DatasetCatalog.register(
        name, lambda: detectron2.data.datasets.coco.load_coco_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

def load_owndataset(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []
    #TODO: add annotations
    # detectron2.data.datasets.coco.py load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None)
    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts

# "coco_2017_unlabel": (
#         "coco/unlabeled2017",
#         "coco/annotations/image_info_unlabeled2017.json",
#     ),

metadata = {}
name_train = "coco_2017_train_subset"
name_val = "coco_2017_val_subset"
json_file_train = "./datasets/coco/subset_annotations/instances_train2017_subset.json"
json_file_val = "./datasets/coco/subset_annotations/instances_val2017_subset.json"
image_root_train = "./datasets/coco/train2017/"
image_root_val = "./datasets/coco/val2017/"
register_own_dataset(name_train, metadata, json_file_train, image_root_train)
register_own_dataset(name_val, metadata, json_file_val, image_root_val)
#_root = os.getenv("DETECTRON2_DATASETS", "datasets")
#register_coco_unlabel_plc(_root)

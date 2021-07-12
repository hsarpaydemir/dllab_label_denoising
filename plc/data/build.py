import logging
import numpy as np
import operator
import json
import torch.utils.data
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)
from ubteacher.data.build import *

class LabeledDatasetStorage:
    data = []
    labels = dict()
    dataloader = None

    @classmethod
    def storeFirstLabels(LabeledDatasetStorage):
        for image in LabeledDatasetStorage.data: # gives images -> get boxes -> get labels
            # store basic labels in labels
                name = image["file_name"]
                boxlabels = []
                for box in image["annotations"]:
                    if not box["iscrowd"] == 1:
                        boxlabels.append(box["category_id"])
                LabeledDatasetStorage.labels[name] = boxlabels
        print("processed labels")

    @classmethod
    def updateLabels(LabeledDatasetStorage, new_labels, label_map):
        counter = 0
        for i, nr_labels in label_map:
            temp_labels = []
            for n in range(nr_labels):
                temp_labels.append(new_labels[counter])
                counter = counter + 1
            LabeledDatasetStorage.labels[i] = temp_labels

        # for i in LabeledDatasetStorage.data:
        #     temp_labels = []
        #     for j in LabeledDatasetStorage.labels[i['file_name']]:
        #         temp_labels.append(new_labels[counter])
        #         counter = counter + 1
        #     LabeledDatasetStorage.labels[i['file_name']] = temp_labels

    @classmethod
    def build_data_loader(LabeledDataStorage, cfg):
        label_dataset = DatasetFromList(LabeledDataStorage.data)
        mapper = DatasetMapper(is_train = True, augmentations=[], image_format=cfg.INPUT.FORMAT)
        label_dataset = MapDataset(label_dataset, mapper)

        #dm = DatasetMapper(,is_train = False)
        LabeledDataStorage.dataloader = torch.utils.data.DataLoader(
            label_dataset,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        #print(dataset)

    @classmethod
    def getDatasetIter(LabeledDatasetStorage):
        return iter(LabeledDatasetStorage.dataloader)


def subset(set):
    #return set[:int(len(set)]
    return set

def add_noise(set):
    return set

def divide_label_unlabel_subset(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    # need to take own split not the given onw
    #num_all = len(dataset_dicts)
    #num_label = int(SupPercent / 100.0 * num_all)

    # # read from pre-generated data seed
    # with open(random_data_seed_path) as COCO_sup_file:
    #     coco_random_idx = json.load(COCO_sup_file)
    # # replace this one
    # labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    # assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    num_all = len(dataset_dicts)
    sup_p = 50
    num_label = int(sup_p / 100. * num_all)
    labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)


    #exit()
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])

    return label_dicts, unlabel_dicts


#uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops_subset(cfg, mapper=None):
    # TODO: loading is here
    #print(cfg.DATASETS.TRAIN)
    #print(DatasetCatalog.get("coco_2017_train"))
    #exit()
    #exit()
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        #print("after loading :",len(dataset_dicts))

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel_subset(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )

        # ugly way to get dicts and add noise
        LabeledDatasetStorage.data = label_dicts.copy()


    # label_dicts are lists. every element is one dictionary with dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations'])
    # needs to have format  dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations'])
    # currently has dict_keys(['file_name', 'height', 'width', 'image_id'])
    # print("Debug :")
    # print("unlabeled : " ,unlabel_dicts[0].keys())
    # print("labeled : " ,label_dicts[0].keys())
    # print(label_dicts[0]["file_name"])
    # exit()

    # TODO: drop images and add noise here

    label_dicts = subset(label_dicts)
    unlabel_dicts = subset(unlabel_dicts)

    label_dicts = add_noise(label_dicts)

    # print("mapping...")
    # print(label_dicts[0].keys())
    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)
    # print(label_dataset[0].keys())
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    # # TODO: check this for mapping
    # print(label_dataset)
    # exit()
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)
    # print(label_dataset[0][0].keys())
    # exit()

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )



# TODO: this is the loader function which returns data (uses iter)
# batch data loader
def build_semisup_batch_data_loader_two_crop_subset(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")

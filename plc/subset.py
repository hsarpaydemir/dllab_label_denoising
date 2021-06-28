import os
import json
import random

def makeSubset(train = True):
    if train:
        path = "../datasets/coco/annotations/instances_train2017.json"
        outpath = "../datasets/coco/subset_annotations/instances_train2017_subset.json"
    else:
        path = "../datasets/coco/annotations/instances_val2017.json"
        outpath = "../datasets/coco/subset_annotations/instances_val2017_subset.json"

    file = open(path)
    json_file = json.load(file)
    print(json_file.keys())
    #print(json_file["images"])
    s = set()
    # exit()
    # for i in range(100):
    #     print(json_file["annotations"][i]["image_id"])
    #     if json_file["annotations"][i]["image_id"] in s:
    #         print("duplicate")
    #     s.add(json_file["annotations"][i]["image_id"])

    if train:
        ratio =0.1
    else:
        ratio = 0.5
    # go though json_file["images"] and select the ones i want
    keeplist = set()
    images = []
    random.seed(0)
    for i in range(len(json_file["images"])):
        if random.random()>(1-ratio):
            keeplist.add(json_file["images"][i]["id"])
            images.append(json_file["images"][i])
    # print(keeplist)
    # print(images)

    annotations = []
    #print(json_file["annotations"])
    #print(json_file["images"][0])
    #print(len(keeplist))
    #print(keeplist)
    #exit()

    for i in range(len(json_file["annotations"])):
        if json_file["annotations"][i]['image_id'] in keeplist:
            annotations.append(json_file["annotations"][i])

    print("keeping ", len(annotations) , " annotations for ", len(keeplist), " images")
    with open(outpath, "w") as dump_file:
        json_file["images"] = images
        json_file["annotations"] = annotations
        json.dump(json_file,dump_file)


if __name__=="__main__":
    makeSubset(train=True)
    makeSubset(train=False)
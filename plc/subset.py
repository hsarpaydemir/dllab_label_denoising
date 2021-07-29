import os
import json
import random
import pdb

def makeSubset(train = True, noise= False):
    if train:
        path = "../datasets/coco/annotations/instances_train2017.json"
        path = os.path.join(os.path.dirname(__file__), path)
        outpath = "../datasets/coco/subset_annotations/instances_train2017_subset.json"
        outpath = os.path.join(os.path.dirname(__file__), outpath)
    else:
        path = "../datasets/coco/annotations/instances_val2017.json"
        path = os.path.join(os.path.dirname(__file__), path)
        outpath = "../datasets/coco/subset_annotations/instances_val2017_subset.json"
        outpath = os.path.join(os.path.dirname(__file__), outpath)

    file = open(path)
    json_file = json.load(file)
    print(json_file.keys())
    pdb.set_trace()
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
        ratio = 0.05
    # go though json_file["images"] and select the ones i want
    keeplist = set()
    images = []
    random.seed(0)
    for i in range(len(json_file["images"])):
        if random.random()>(1-ratio) or noise:
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


def makeNoisySet(train=True):
    if train:
        path = "../datasets/coco/subset_annotations/instances_train2017_subset.json"
        path = os.path.join(os.path.dirname(__file__), path)
        outpath = "../datasets/coco/noisy_annotations/instances_train2017_10%.json"
        outpath = os.path.join(os.path.dirname(__file__), outpath)
    else:
        path = "../datasets/coco/subset_annotations/instances_val2017_subset.json"
        path = os.path.join(os.path.dirname(__file__), path)
        outpath = "../datasets/coco/noisy_annotations/instances_val2017.json"
        outpath = os.path.join(os.path.dirname(__file__), outpath)

    print("opening file")
    file = open(path)
    json_file = json.load(file)
    print(json_file.keys())
    threshold = 0.80

    tmp_arr = []
    for i in json_file['categories']:
        tmp_arr.append(i['id'])

    random.seed(173489755)
    for i in json_file['annotations']:
        if random.random()>=threshold:
            while True:
                rand_tmp = random.randint(1, 90)
                if rand_tmp in tmp_arr:
                    i['category_id'] = rand_tmp
                    break

    '''
    for i in json_file['annotations']:
        if random.random()>=threshold:
            while True:
                rand_tmp = random.randint(1, 90)
                if rand_tmp not in random_picked or (len(random_picked) >= 90):
                    i['category_id'] = rand_tmp
                    random_picked.append(rand_tmp)
                    break
    '''

    outfile = open(outpath, "w")
    json.dump(json_file, outfile)


if __name__=="__main__":
    makeSubset(train=True)
    #makeSubset(train=False)
    #makeNoisySet()
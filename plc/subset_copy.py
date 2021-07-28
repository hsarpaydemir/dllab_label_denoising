import os
import json
import random
import pdb
import numpy as np
from tqdm import tqdm

NUMBER_OF_CLASSES = 30

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
    
    #print(json_file["images"])
    
    top_classes = find_top_classes()

    if train:
        ratio =0.05
    else:
        ratio = 0.05
    # go though json_file["images"] and select the ones i want
    keeplist = set()
    images = []
    random.seed(0)

    img_bbxs = {}
    for annotation in json_file['annotations']:
        if annotation['image_id'] in img_bbxs:
            img_bbxs[annotation['image_id']].append(annotation['category_id'])
        else:
            img_bbxs[annotation['image_id']] = [annotation['category_id']]

    for i in range(len(json_file["images"])):
        try:
            sub_list = img_bbxs[json_file['images'][i]['id']]
            if (random.random()>(1-ratio) or noise) and set(sub_list).issubset(set(top_classes)):
                keeplist.add(json_file["images"][i]["id"])
                images.append(json_file["images"][i])
        except:
            continue
    pdb.set_trace()
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

def find_top_classes(train = True, noise= False):
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
    
    #print(json_file["images"])
    frequencies = [0] * 91
    for annotation in json_file['annotations']:
        frequencies[annotation['category_id']] += 1

    top_classes = np.flip(np.array(frequencies).argsort()[-NUMBER_OF_CLASSES:])
    for i in top_classes:
        print("Number of classes in category {} is {}".format(i, frequencies[i]))
    
    return top_classes.tolist()

if __name__=="__main__":
    #find_top_classes(train=True)
    makeSubset(train=True)
    #makeSubset(train=False)
    #makeNoisySet()
    
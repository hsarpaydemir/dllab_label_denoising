import os
import json
import random
import pdb
import numpy as np
from tqdm import tqdm

NUMBER_OF_CLASSES = 10

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
        ratio = 0.4
    # go though json_file["images"] and select the ones i want
    keeplist = set()
    images = []
    random.seed(0)

    keeplist = set()
    images = []
    random.seed(0)
    for i in tqdm(range(len(json_file["images"]))):
        if random.random()>(1-ratio) or noise:
            keeplist.add(json_file["images"][i]["id"])
            #images.append(json_file["images"][i])

    new_annotations = []
    new_image_ids = []
    new_images = []
    for annotation in tqdm(json_file['annotations']):
        if (annotation['image_id'] in keeplist) and (annotation['category_id'] in top_classes):
            new_annotations.append(annotation)
            if annotation['image_id'] not in new_image_ids:
                new_image_ids.append(annotation['image_id'])

    for image in tqdm(json_file['images']):
        if image['id'] in new_image_ids:
            new_images.append(image)
    
    '''
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
    '''
    # print(keeplist)
    # print(images)

    annotations = []
    #print(json_file["annotations"])
    #print(json_file["images"][0])
    #print(len(keeplist))
    #print(keeplist)
    #exit()

    '''
    for i in range(len(json_file["annotations"])):
        if json_file["annotations"][i]['image_id'] in keeplist:
            annotations.append(json_file["annotations"][i])
    '''

    print("keeping ", len(new_annotations) , " annotations for ", len(new_images), " images")
    with open(outpath, "w") as dump_file:
        json_file["images"] = new_images
        json_file["annotations"] = new_annotations
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

    top_classes = find_top_classes()
    random.seed(173489755)
    for i in json_file['annotations']:
        if random.random()>=threshold:
            i['category_id'] = random.choice(top_classes)

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
    #makeSubset(train=True)
    makeSubset(train=False)
    #makeNoisySet()
    
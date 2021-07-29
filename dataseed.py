import json
import random

original="./dataseed/coco_supervision.txt"
def whatistheseed():
    with open(original,"r") as seedfile:
        j = json.load(seedfile)
    print(j.keys())
    for i in j.keys():
        for ls in j[i].keys():
            print(len(j[i][ls]))

def add20percent():
    # get all file names
    with open("./datasets/coco/annotations/instances_train2017.json") as annotations:
        # collect all image names
        j = json.load(annotations)

    image_set = set()

    for i in j['annotations']:
        if i['image_id'] not in image_set:
            image_set.add(i['image_id'])

    image_set = list(image_set)
    seedlist=[]
    accepted = set()

    #was 2927 before
    while len(accepted)<2352:
        r = random.randint(0,len(image_set)-1)
        if image_set[r] not in accepted:
            accepted.add(image_set[r])

    d = dict()
    seedlist = list(accepted)
    original = "./dataseed/COCO_supervision_20.txt"
    d['20.0'] = dict()
    d['20.0']['1']=seedlist
    with open(original, "w") as seedfile:
        json.dump(d, seedfile)

if __name__=="__main__":
    #whatistheseed()
    add20percent()
    pass
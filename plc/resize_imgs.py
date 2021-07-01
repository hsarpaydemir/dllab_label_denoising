from PIL import Image
import os
import json
import pdb
import copy
from tqdm import tqdm

DOWNSCALE_FACTOR = 8

directory = "../datasets/coco/train2017/"
directory = os.path.join(os.path.dirname(__file__), directory)

out_directory = "../datasets/cocoresized/train2017/"
out_directory = os.path.join(os.path.dirname(__file__), out_directory)

for filename in tqdm(os.listdir(directory)):
    img_full = directory + filename
    im = Image.open(img_full, mode = 'r')
    width, height = im.size
    im = im.resize([width // DOWNSCALE_FACTOR, height // DOWNSCALE_FACTOR], Image.NEAREST)
    im.save(out_directory + filename)


directory = "../datasets/coco/val2017/"
directory = os.path.join(os.path.dirname(__file__), directory)

out_directory = "../datasets/cocoresized/val2017/"
out_directory = os.path.join(os.path.dirname(__file__), out_directory)

for filename in tqdm(os.listdir(directory)):
    img_full = directory + filename
    im = Image.open(img_full, mode = 'r')
    width, height = im.size
    im = im.resize([width // DOWNSCALE_FACTOR, height // DOWNSCALE_FACTOR], Image.NEAREST)
    im.save(out_directory + filename)

path = "datasets/coco/annotations/instances_train2017.json"
file = open(path)
json_file = json.load(file)
final_json = json_file

for img in tqdm(json_file['annotations']):
    bounding_box = copy.deepcopy(img['bbox'])
    img['bbox'][0] = bounding_box[0] // DOWNSCALE_FACTOR
    img['bbox'][1] = bounding_box[1] // DOWNSCALE_FACTOR
    img['bbox'][2] = bounding_box[2] // DOWNSCALE_FACTOR
    img['bbox'][3] = bounding_box[3] // DOWNSCALE_FACTOR
    #print('BeforeBbox: ', bounding_box, '\tAfterBbox :', img['bbox'], '\n')

for img in tqdm(json_file['images']):
    img_dims = (img['width'], img['height'])
    img['width'] = img['width'] // DOWNSCALE_FACTOR
    img['height'] = img['height'] // DOWNSCALE_FACTOR
    #print('BeforeDims: ', img_dims, '\t AfterDims', (img['width'], img['height']), '\n')

outpath = 'datasets/annotations_new/instances_train2017.json'
with open(outpath, "w") as dump_file:
        json.dump(json_file,dump_file)

path = "datasets/coco/annotations/instances_val2017.json"
file = open(path)
json_file = json.load(file)
final_json = json_file

for img in tqdm(json_file['annotations']):
    bounding_box = copy.deepcopy(img['bbox'])
    img['bbox'][0] = bounding_box[0] // DOWNSCALE_FACTOR
    img['bbox'][1] = bounding_box[1] // DOWNSCALE_FACTOR
    img['bbox'][2] = bounding_box[2] // DOWNSCALE_FACTOR
    img['bbox'][3] = bounding_box[3] // DOWNSCALE_FACTOR
    #print('BeforeBbox: ', bounding_box, '\tAfterBbox :', img['bbox'], '\n')

for img in tqdm(json_file['images']):
    img_dims = (img['width'], img['height'])
    img['width'] = img['width'] // DOWNSCALE_FACTOR
    img['height'] = img['height'] // DOWNSCALE_FACTOR
    #print('BeforeDims: ', img_dims, '\t AfterDims', (img['width'], img['height']), '\n')

outpath = 'datasets/annotations_new/instances_val2017.json'
with open(outpath, "w") as dump_file:
        json.dump(json_file,dump_file)
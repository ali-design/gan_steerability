import urllib

IMAGENET_DIR = '/mnt/disks/sdb/imagenet/train' # path to imagenet ims locally
IMAGENET_URL = 'https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt'

def get_wid_imagenet():
    content=urllib.request.urlopen(IMAGENET_URL)
    wids = []
    category_names = []
    for ii, line in enumerate(content):
        (wid, label) = line.decode('ascii').strip().split(" ", 1)
        wids.append((ii, wid))
        category_names.append(label.split(',', 1)[0])
    return wids

def get_category_names():
    content=urllib.request.urlopen(IMAGENET_URL)
    category_names = []
    for ii, line in enumerate(content):
        (wid, label) = line.decode('ascii').strip().split(" ", 1)
        category_names.append(label.split(',', 1)[0])
    return category_names

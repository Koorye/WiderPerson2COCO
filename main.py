# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date:   2022-4-10
import json
import os
from PIL import Image
from tqdm import tqdm


root = '/home/koorye/datasets/WiderPerson/'
ann_path_template = 'instances_{}.json'

clses = ['pedestrains', 'riders', 'paritally-visible persons', 'ignore regions', 'crowd']

def get_img_ids(path):
    with open(path) as f:
        s = f.readlines()
        s = list(map(lambda x: x.strip(), s))
    return s

def get_cats():
    cats = []
    for id_, cls in enumerate(tqdm(clses, desc='Getting Categories')):
        cats.append(dict(
            id=id_,
            name=cls,
        ))
    return cats

def get_anns(img_ids):
    anns = []
    id_ = 0

    for img_id in tqdm(img_ids, desc='Getting Annotations'):
        with open(os.path.join(root, f'Annotations/{img_id}.jpg.txt')) as f:
            s = f.readlines()
            for i in range(len(s)):
                if i == 0:
                    continue
                
                line = s[i]
                cls_id, x1, y1, x2, y2 = line.strip().split(' ')
                cls_id = int(cls_id) - 1
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                w, h = x2 - x1, y2 - y1
                
                area = w * h
                iscrowd = 1 if cls_id == 4 else 0
                bbox = [x1, y1, w, h]
        
                anns.append(dict(
                    area=area,
                    iscrowd=iscrowd,
                    image_id=int(img_id),
                    bbox=bbox,
                    category_id=cls_id,
                    id=id_,
                ))
        
                id_ += 1
    
    return anns  

def get_imgs(img_ids):
    imgs = []
    for img_id in tqdm(img_ids, desc='Getting Images'):
        img_name = f'{img_id}.jpg'

        img_path = os.path.join(root, 'Images/', img_name)
        img = Image.open(img_path)
        w, h = img.size
        
        imgs.append(dict(
            file_name=img_name,
            height=h,
            width=w,
            id=int(img_id)
        ))

    return imgs

def get_ann_dict(img_ids):
    cats = get_cats()
    anns = get_anns(img_ids)
    imgs = get_imgs(img_ids)
    
    return dict(
        images=imgs,
        annotations=anns,
        categories=cats,
    )

def save_ann_file(ann_dict, name, min=False):
    path = os.path.join(root, 'Annotations/', ann_path_template.format(name))
    print('Annotation file will be saved to:', path)
    
    with open(path, 'w') as f:
        if min:
            json.dump(ann_dict, f)
        else:
            json.dump(ann_dict, f, indent=2)

train_img_ids = get_img_ids(os.path.join(root, 'train.txt'))
val_img_ids = get_img_ids(os.path.join(root, 'val.txt'))

train_ann = get_ann_dict(train_img_ids)
val_ann = get_ann_dict(val_img_ids)

save_ann_file(train_ann, 'train')
save_ann_file(val_ann, 'val')

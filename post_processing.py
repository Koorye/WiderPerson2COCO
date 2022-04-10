import json
import os
import pandas as pd
from tqdm import tqdm


convert_id_dict = {
    0: 0,
    1: 0,
    2: None,
    3: None,
    4: None,
}

root = '/home/koorye/datasets/WiderPerson'
ann_path_template = 'instances_{}.json'
ann_file_names = ['train', 'val']

def dict_drop_duplicate(dict):
    return pd.DataFrame(dict).drop_duplicates().to_dict(orient='records')

for file_name in ann_file_names:
    path = os.path.join(root, 'Annotations/', ann_path_template.format(file_name)) 
    
    with open(path) as f:
        ann_dict = json.load(f)   
        imgs = ann_dict['images']
        anns = ann_dict['annotations']
        cats = ann_dict['categories']
    
    id2name = {cat['id']: cat['name'] for cat in cats}
    
    for idx, cat in enumerate(tqdm(cats, desc='Converting Categories')):
        if convert_id_dict[cat['id']] is not None:
            cat['id'] = convert_id_dict[cat['id']]
            cat['name'] = id2name[cat['id']]
        else:
            cats[idx] = None
        
    cats = list(filter(lambda x: x is not None, cats))
    cats = dict_drop_duplicate(cats)

    for idx, ann in enumerate(tqdm(anns, desc='Converting Annotaions')):
        new_cat_id = convert_id_dict[ann['category_id']]
        if new_cat_id is None:
            anns[idx] = None
        else:
            ann['category_id'] = new_cat_id
        
    anns = list(filter(lambda x: x is not None, anns)) 
    for idx, ann in enumerate(tqdm(anns, desc='Reseting Annotation ID')):
        ann['id'] = idx
   
    save_path = os.path.join(root, 'Annotations/', ann_path_template.format(file_name+'_posted')) 

    new_ann_dict = dict(
        images=imgs,
        annotations=anns,
        categories=cats,
    )
    with open(save_path, 'w') as f:
        print('Annotation files will be saved to:', save_path)
        json.dump(new_ann_dict, f, indent=2)

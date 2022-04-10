import cv2
import random
import os
from pycocotools.coco import COCO


img_id = None
root = '/home/koorye/datasets/WiderPerson'
ann_file_name = 'instances_val_posted.json'
scale = 2.

coco = COCO(annotation_file=os.path.join(root, 'Annotations/', ann_file_name))

cat_ids = coco.getCatIds()
img_ids = coco.getImgIds()

if img_id is None:
    img_id = img_ids[0]

ann_ids = coco.getAnnIds([img_id])
cats = coco.loadCats(cat_ids)
img = coco.loadImgs([img_id])[0]
anns = coco.loadAnns(ann_ids)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
          for _ in range(len(cats))]

img_path = os.path.join(root, 'Images/', img['file_name'])
img = cv2.imread(img_path)
x, y = img.shape[0:2]
img = cv2.resize(img, (int(y*scale), int(x*scale)))

def draw_box(img, pt1, pt2, cls_id):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    img = cv2.rectangle(img, pt1, pt2, colors[cls_id], 2)
    pt1 = (pt1[0], pt1[1] + 10)
    img = cv2.putText(img, cats[cls_id]['name'], pt1, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return img

for ann in anns:
    xmin, ymin, w, h = ann['bbox']
    cls_id = ann['category_id']
    xmax, ymax = xmin + w, ymin + h
    
    xmin *= scale
    ymin *= scale
    xmax *= scale
    ymax *= scale
    
    img = draw_box(img, (xmin, ymin), (xmax, ymax), cls_id)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

import os
import time
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 

from config import Config
import utils

ROOT_DIR = os.getcwd()
ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'data/class_annotations.csv')

class OcrConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ocr"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    
    DETECTION_MAX_INSTANCES = 180
    
class OcrDataset(utils.Dataset):
    
    def load_ocr(self, dataset_np, img_base_path, json_base_path, width, height):
        self.add_class("ocr", 1, "box")
        # self.add_class("ocr", 2, "o")
        # self.add_class("ocr", 3, "b")
        self.class_mapping = {
            "product_id": 2,
            "order_num": 3,
            "box": 1,
            "boxv": 1,
        }
        self.img_base_path = img_base_path
        self.json_base_path = json_base_path
        for idx, p in enumerate(dataset_np):
            self.add_image('ocr',
                           image_id=idx,
                           path=os.path.join(img_base_path, p),
                           img_name=p,
                           json_name="%s.json" % os.path.splitext(os.path.basename(p))[0],
                           width=width, height=height)
#         self.dataset_np = np.load(dataset_np)
        
#         self.df = pd.read_csv(annotations, names = ["name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "classname"])

#         for idx, p in enumerate(self.df.name.unique()):
#             print(p)
#             dataframe = self.df.loc[self.df['name'] == p]
#             print(dataframe.shape)
#             self.add_image('ocr',
#                            image_id=idx,
#                            path=os.path.join(base_path, p),
#                            img_name=p,
#                            width=width, height=height,
#                            dataframe=dataframe
#                            )
    
    def load_image(self, image_id):
        path = "{}".format(self.image_info[image_id]['path'])
        # print("load_image: ", path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        json_name = info['json_name']
        # print(json_name)
        instance_masks = []
        class_ids = []

        img = np.zeros([info['height'], info['width']], dtype=np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df = pd.read_json(os.path.join(self.json_base_path, json_name), typ='series')
        
        # print(df)
        for idx, shape in enumerate(df.shapes):
            class_id = self.class_mapping[shape['label']]
            if class_id == 1:
                # print(shape['points'])
                box = shape['points']
                m = utils.draw_polygon(img.copy(), box, 1)
                instance_masks.append(m)
                class_ids.append(class_id)
            # if idx > 20:
            #     break
                
        
#         for idx, row in info['dataframe'].iterrows():
#             class_id = self.class_mapping[row.classname]
#             if class_id:
#                 box = row[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values.reshape(-1)
#                 m = utils.draw_polygon(img.copy(), box, 1)
#                 instance_masks.append(m)
#                 class_ids.append(class_id)
        
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
    

    
if __name__ == '__main__':
    import argparse
        
       
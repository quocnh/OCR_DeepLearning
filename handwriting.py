import os
import time
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 

from config import Config
import utils

ROOT_DIR = os.getcwd()
ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'data/class_annotations.csv')

class HandwritingConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "handwriting"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 10

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # background + 3 classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    
    # MINI_MASK_SHAPE = (8, 8) 
    
    # MASK_SHAPE = [8, 8]
    
    DETECTION_MAX_INSTANCES = 10
    
    MAX_GT_INSTANCES = 10
    
class HandwritingDataset(utils.Dataset):
    
    def load_dataset_old(self, dataset_np, img_base_path, json_base_path, width, height):
        self.add_class("handwriting", 1, "one")
        self.add_class("handwriting", 2, "two")
        self.add_class("handwriting", 3, "three")
        self.add_class("handwriting", 4, "four")
        self.add_class("handwriting", 5, "five")
        self.add_class("handwriting", 6, "six")
        self.add_class("handwriting", 7, "seven")
        self.add_class("handwriting", 8, "eight")
        self.add_class("handwriting", 9, "nine")
        self.add_class("handwriting", 10, "zero")
        self.class_mapping = {
            "0": 10,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
        }
        self.img_base_path = img_base_path
        self.json_base_path = json_base_path
        for idx, p in enumerate(dataset_np):
            # print(p)
            self.add_image('handwriting',
                           image_id=idx,
                           path=os.path.join(img_base_path, p),
                           img_name=p,
                           json_name="%s.json" % os.path.splitext(os.path.basename(p))[0],
                           width=width, height=height)
    def load_dataset(self, images_path, masks_path, labels_path):
        self.add_class("handwriting", 1, "one")
        self.add_class("handwriting", 2, "two")
        self.add_class("handwriting", 3, "three")
        self.add_class("handwriting", 4, "four")
        self.add_class("handwriting", 5, "five")
        self.add_class("handwriting", 6, "six")
        self.add_class("handwriting", 7, "seven")
        self.add_class("handwriting", 8, "eight")
        self.add_class("handwriting", 9, "nine")
        self.add_class("handwriting", 10, "zero")
        images = np.load(images_path)
        masks = np.load(masks_path)
        labels = np.load(labels_path)
        for idx, img in enumerate(images):
            # print(p)
            self.add_image('handwriting',
                           image_id=idx,
                           path = idx,
                           image = images[idx],
                           masks = masks[idx],
                           labels = labels[idx])
    
    def load_image_old(self, image_id):
        path = "{}".format(self.image_info[image_id]['path'])
        # print("load_image: ", path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_image(self, image_id):
        image = self.image_info[image_id]['image']
        return image
    
    def load_mask_old(self, image_id):
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
            if class_id:
                # print(shape['points'])
                box = shape['points']
                m = utils.draw_polygon(img.copy(), box, 1)
                instance_masks.append(m)
                class_ids.append(class_id)
        
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
    
    def load_mask(self, image_id):
        # print("load_mask")
        info = self.image_info[image_id]
        masks = info['masks']
        labels = info['labels']
        # print(masks.shape)
        # print(labels.shape)
        instance_masks = []
        class_ids = []
        
        for idx, label in enumerate(labels):
            # print("-----", label)
            if not label > 9:
                # print("------", masks[idx].shape)
                instance_masks.append(masks[idx])
                if label == 0:
                    class_ids.append(10)
                else:
                    class_ids.append(label)
        
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            # print("----------------mask", mask.shape)
            # print("----------------class_ids", class_ids.shape)
            return mask, class_ids

    
if __name__ == '__main__':
    import argparse
        
       
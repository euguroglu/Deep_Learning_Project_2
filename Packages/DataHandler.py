import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

food_classes = ['bread','dairy_product','dessert','egg','fried_food','meat','noodles_pasta',
               'rice','seafood','soup','vegetable']

def split_data_into_class_folders(path_to_data,class_id):
    imgs_paths = glob.glob(path_to_data + '*.jpg')

    for path in imgs_paths:

        basename = os.path.basename(path)

        if basename.startswith(str(class_id)+'_'):

            path_to_save = os.path.join(path_to_data,food_classes[class_id])

            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)

            shutil.move(path,path_to_save)


def visualize_some_image(path_to_data):

    imgs_paths = []
    labels = []

    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                imgs_paths.append(os.path.join(r,file))
                labels.append(os.path.basename(r))

    fig = plt.figure()

    for i in range(16):
        chosen_index = random.randint(0,len(imgs_paths)-1)
        chosen_img = imgs_paths[chosen_index]
        chosen_label = labels[chosen_index]

        ax = fig.add_subplot(4,4,i+1)
        ax.title.set_text(chosen_label)
        ax.imshow(Image.open(chosen_img))
    plt.show()


if __name__=='__main__':

    split_data_switch = False
    visualize_data_switch = True

    path_to_eval_data = 'C:/Users/eugur/Deep_Learning_Deployment/food-11/evaluation/'
    path_to_train_data = 'C:/Users/eugur/Deep_Learning_Deployment/food-11/training/'
    path_to_val_data = 'C:/Users/eugur/Deep_Learning_Deployment/food-11/validation/'

    if split_data_switch:
        for i in range(11):
            split_data_into_class_folders(path_to_train_data,i)
        for i in range(11):
            split_data_into_class_folders(path_to_eval_data,i)
        for i in range(11):
            split_data_into_class_folders(path_to_val_data,i)

    if visualize_some_image:
        visualize_some_image(path_to_train_data)

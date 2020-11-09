import glob
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import storage

path_to_credentials = './credentials/infra-actor-295022-b2f5aab4b230.json'

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
    fig.tight_layout(pad=0.05)
    plt.show()


def get_images_sizes(path_to_data):

    imgs_paths = []
    widths = []
    heights = []

    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(r,file))
                widths.append(img.size[0])
                heights.append(img.size[1])
                img.close()

    mean_widths = sum(widths)/len(widths)
    mean_heights = sum(heights)/len(heights)
    median_widths = np.median(widths)
    median_heights = np.median(heights)

    return mean_widths,mean_heights,median_widths,median_heights

def list_blobs(bucket_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)

    blobs = storage_client.list_blobs(bucket_name)

    return blobs




if __name__=='__main__':

    split_data_switch = False
    visualize_data_switch = False
    print_insight_switch = False
    list_blobs_switch = True

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

    if visualize_data_switch:
        visualize_some_image(path_to_train_data)


    if print_insight_switch:
        mean_widths,mean_heights,median_widths,median_heights = get_images_sizes(path_to_train_data)
        print('Mean width: {}'.format(mean_widths))
        print('Mean height: {}'.format(mean_heights))
        print('Median width: {}'.format(median_widths))
        print('Median height: {}'.format(median_heights))

    if list_blobs_switch:
        blobs = list_blobs('dummy-bucket-food-dataset')

        for blob in blobs:
            print(blob.name)

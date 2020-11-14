from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from DataHandler import download_data_to_local_directory, upload_data_to_bucket
from tensorflow.python.client import device_lib
import argparse
import hypertune
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import shutil
from datetime import datetime

print("Tensorflow is running on following devices: ")
print(device_lib.list_local_devices())


def build_model(nbr_classes):

    base_model = InceptionV3(weights="imagenet",include_top=False,input_tensor=Input(shape=(229,229,3)))

    head_model = base_model.output

    head_model = Flatten()(head_model)
    head_model = Dense(512,activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(nbr_classes, activation="softmax")(head_model)

    model = Model(inputs=base_model.input,outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    return model


def build_data_pipelines(batch_size,train_data_path,val_data_path,eval_data_path):

    train_augmentor = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 25,
        zoom_range = 0.15,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.15,
        horizontal_flip = True,
        fill_mode ='nearest'
    )

    val_augmentor = ImageDataGenerator(
        rescale = 1./255
    )

    train_generator = train_augmentor.flow_from_directory(
        train_data_path,
        class_mode='categorical',
        target_size = (229,229),
        color_mode = "rgb",
        shuffle = True,
        batch_size = batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode='categorical',
        target_size = (229,229),
        color_mode = "rgb",
        shuffle = False,
        batch_size = batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode='categorical',
        target_size = (229,229),
        color_mode = "rgb",
        shuffle = False,
        batch_size = batch_size
    )
    return train_generator,val_generator,eval_generator

def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for root,dirnames,files in os.walk(directory):
        for file in files:
            _,ext = os.path.splitext(file)
            if ext in ['.png','.jpg','jpeg']:
                totalcount = totalcount + 1
    return totalcount


def train(path_to_data,batch_size,epochs,learning_rate,models_bucket_name):

    path_train_data = os.path.join(path_to_data,'training')
    path_val_data = os.path.join(path_to_data,'validation')
    path_eval_data = os.path.join(path_to_data,'evaluation')

    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)

    print(total_train_imgs,total_val_imgs,total_eval_imgs)


    train_generator,val_generator,eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data
    )

    classes_dict = train_generator.class_indices
    model = build_model(nbr_classes=len(classes_dict.keys()))

    optimizer = Adam(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor="val_loss",patience=10)

    path_to_save_model = "./tmp"
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    checkpoint_saver = ModelCheckpoint(path_to_save_model,
                                        monitor="val_accuracy",
                                        mode="max",
                                        save_best_only=True,
                                        save_freq="epoch",
                                        verbose=1)

    model.fit_generator(
        train_generator,
        steps_per_epoch= total_train_imgs // batch_size,
        validation_data=val_generator,
        validation_steps= total_val_imgs // batch_size,
        epochs = epochs,
        callbacks=[early_stopping,checkpoint_saver]
    )

    print('[INFO] Evaluation phase...')

    predictions = model.predict_generator(eval_generator)
    predictions_idxs = np.argmax(predictions,axis=1)
    my_classification_report = classification_report(eval_generator.classes,predictions_idxs,target_names=eval_generator.class_indices.keys())
    my_confusion_matrix = confusion_matrix(eval_generator.classes,predictions_idxs)

    print("[INFO] Classification report:")
    print(my_classification_report)
    print("[INFO] Confusion matrix")
    print(my_confusion_matrix)

    print("Starting evaluation using model.evaluate_generator")
    scores = model.evaluate_generator(eval_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hyptertune = {loss}")

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    zipped_folder_name = f"trained_model_{now}_loss_{loss}"
    shutil.make_archive(zipped_folder_name,"zip","/usr/src/app/tmp")

    path_zipped_folder =  "/usr/src/app/"+zipped_folder_name+".zip"
    upload_data_to_bucket(models_bucket_name,path_zipped_folder,zipped_folder_name)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss',
                                            metric_value=loss, global_step=epochs)



if __name__ =="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket_name",type=str, help="Bucket name on Google cloud storage",
                        default='enes-data-bucket')
    parser.add_argument("--models_bucket_name",type=str, help="Bucket name on Google cloud storage for saving trained models",
                        default='trained_models_food_classification_bucket')
    parser.add_argument("--batch_size",type=int, help="Batch size used by the deep learning model",
                        default=2)
    parser.add_argument("--learning_rate",type=float, help="Learning rate used by the deep learning model",
                        default=1e-5)


    args = parser.parse_args()

    print("Data Download started...")
    download_data_to_local_directory(args.bucket_name,'C:/Users/eugur/Deep_Learning_Deployment/data')
    print('Download finished...')
    path_to_data = 'C:/Users/eugur/Deep_Learning_Deployment/data'
    train(path_to_data,args.batch_size,20,args.learning_rate,args.models_bucket_name)

from tensorflow.keras.preprocessing import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizer import SGD,Adam

def build_model(nbr_classes):

    base_model = InceptionV3(weights="imagenet",include_top=False,input_tensor=Input(shape=(229,229,3)))

    head_model = base_model.output

    head_model = Flatten()(head_model)
    head_model = Dense(512)(head_model)
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
        height_shift_Range = 0.2,
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
        color_mode = "rgb"
        shuffle = True,
        batch_size = batch_size
    )

    val_generator = val_augmentor.flow_from_directory(
        val_data_path,
        class_mode='categorical',
        target_size = (229,229),
        color_mode = "rgb"
        shuffle = False,
        batch_size = batch_size
    )

    eval_generator = val_augmentor.flow_from_directory(
        eval_data_path,
        class_mode='categorical',
        target_size = (229,229),
        color_mode = "rgb"
        shuffle = False,
        batch_size = batch_size
    )
    return train_generator,val_generator,eval_generator

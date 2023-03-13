import cv2
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, save_img
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import tensorflow_addons as tfa
import argparse
import random as rn
import numpy as np
import glob
import matplotlib.pyplot as plt
from imutils import paths
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa
# from tensorflow.keras import utils
# ---import spp --------

# -----Fix Cudnn status internal error with RTX series GPU and tensorflow 1.15
# although it might be a general problem with CUDA 10.0,
# since the new cards don't support the older versions)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# -----Fix Cudnn status internal error with RTX series GPU
rn.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
ap = argparse.ArgumentParser()
ap.add_argument("-type", default='test', help="train or test")
args = vars(ap.parse_args())


def TransferVGGmodel():

    # Load model without classifier layers
    model = VGG16(include_top=True, input_shape=(256, 256, 3))

    for layer in model.layers:
        layer.trainable = True
    # add new classifier layers
    # using spp from tensorflow
    # the [-i] last fully connected layer will be the new output layer.
    lastconv = model.layers[-2].output
    spp = tfa.layers.SpatialPyramidPooling2D(
        [1, 2, 4], data_format='channels_last')(lastconv)
    flatten = Flatten()(spp)
    class1 = Dense(1024, activation='relu')(flatten)
    output = Dense(5, activation='softmax')(class1)

    model = Model(inputs=model.inputs, outputs=output)
    model.summary()

    model.compile(
        optimizer=tf.compat.v1.train.MomentumOptimizer(),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        augmentation=iaa.Sometimes(5/6, iaa.OneOf([
            iaa.Fliplr(1),
            iaa.Flipud(1),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Affine(rotate=(-90, 90)),
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Fliplr(0.5),  # 左右翻轉概率為0.5
            iaa.Flipud(0.5),  # 上下翻轉概率為0.5
            iaa.Affine(rotate=(-10, 10)),  # 隨機旋轉-10°到10°
            iaa.Affine(scale=(0.8, 1.2)),  # 隨機縮放80%-120%
            iaa.Crop(percent=(0, 0.1)),  # 隨機裁剪，裁剪比例為0%-10%
            iaa.GaussianBlur(sigma=(0, 0.5)),  # 高斯模糊，sigma值在0到0.5之間
            # 添加高斯噪聲，噪聲標準差為0到0.05的像素值
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
            iaa.ContrastNormalization((0.5, 1.5)),  # 對比度調整，調整因子為0.5到1.5
        ]))
    )
    # Create dataset
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)
    val_datagen = ImageDataGenerator(rescale=1./255,
                                     validation_split=0.1)
    training_set = train_datagen.flow_from_directory('Dataset/FinalMelon/'+'train',
                                                     target_size=(256, 256),
                                                     shuffle=False,
                                                     batch_size=16,
                                                     class_mode='categorical',
                                                     subset='training')

    val_set = val_datagen.flow_from_directory('Dataset/FinalMelon/'+'train',
                                              target_size=(256, 256),
                                              shuffle=False,
                                              batch_size=16,
                                              class_mode='categorical',
                                              subset='validation')

    test_set = test_datagen.flow_from_directory('Dataset/FinalMelon/'+'test',
                                                target_size=(256, 256),
                                                shuffle=False,
                                                batch_size=1,
                                                class_mode='categorical',
                                                )
    callbacks = [tf.keras.callbacks.ModelCheckpoint("best_transferVGG16_SPP_SkinClassification_model.h5", save_best_only=True, monitor="val_loss"),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001)]

    r = model.fit(
        training_set,
        epochs=500,
        steps_per_epoch=804,
        callbacks=callbacks,
        validation_data=val_set

    )
    plt.plot(r.history['accuracy'], label='train acc')
    plt.plot(r.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.savefig('AccVal_best_transferVGG16_SPP_SkinClassification_model.jpg')
    # Evaluate model
    test_set.reset()
    model = tf.keras.models.load_model(
        "best_transferVGG16_SPP_SkinClassification_model.h5")
    test_loss, test_acc = model.evaluate(test_set)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    predicted = model.predict(test_set)
    print("y_pred before ==", predicted)
    # print("y_test before ==", Y_test)
    Y_test = test_set.labels
    y_pred = np.argmax(predicted, axis=1)
    # Y_test = np.argmax(Y_test, axis=1)
    print("y_pred after ==", y_pred)
    print("y_test after ==", Y_test)

    print("Accuracy of VGG16 method: ", accuracy_score(Y_test, y_pred))
    target_names = ['Mixed_color', 'Netting', 'Smooth', 'Spots', 'Wrinkled']
    print(classification_report(Y_test, y_pred,
          target_names=target_names, digits=4))


def testVGG16():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory('Dataset/FinalMelon/'+'test',
                                                target_size=(256, 256),
                                                shuffle=False,
                                                batch_size=1,
                                                class_mode='categorical',
                                                )
    model = tf.keras.models.load_model(
        "best_transferVGG16_SPP_SkinClassification_model.h5")

    predicted = model.predict(test_set)
    Y_test = test_set.labels
    y_pred = np.argmax(predicted, axis=1)

    print("y_pred after ==", y_pred)

    print("Accuracy of VGG16 method: ", accuracy_score(Y_test, y_pred))

    target_names = ['Mixed_color', 'Netting', 'Smooth', 'Spots', 'Wrinkled']
    print(classification_report(Y_test, y_pred,
          target_names=target_names, digits=4))

    input_test = 'Dataset/FinalMelon/test'
    correctPre = 'Dataset/FinalMelon/output/correctPre'
    wrongPre = 'Dataset/FinalMelon/output/wrongPre'

    # Remove files inside output folder
    correctPreFiles = glob.glob(correctPre+'/*')
    wrongPreFiles = glob.glob(wrongPre+'/*')
    for f in correctPreFiles:
        os.remove(f)
    for f in wrongPreFiles:
        os.remove(f)

    count = 0
    output_array = []
    for imagePath in paths.list_images(input_test):
        # print("imagePath==",imagePath)
        realLabel = imagePath.split(os.path.sep)[-2]
        # print("realLabel==",realLabel)
        img = image.load_img(imagePath, target_size=(256, 256))
        ori_img = image.load_img(imagePath)
        img_array = image.img_to_array(img)

        ori_array = image.img_to_array(ori_img)
        img_array = img_array.astype('float32')/255
        img_array = np.expand_dims(img_array, axis=0)

        # predict value of input image
        y = model.predict(img_array)
        y_pre = np.argmax(y, axis=1)
        output_array.append(y_pre[0])
        # covert int value to string label
        if (y_pre[0] == 0):
            preLabel = "Mixed_color"
        if (y_pre[0] == 1):
            preLabel = "Netting"
        if (y_pre[0] == 2):
            preLabel = "Smooth"
        if (y_pre[0] == 3):
            preLabel = "Spots"
        if (y_pre[0] == 4):
            preLabel = "Wrinkled"
        ori_array = cv2.putText(ori_array, "Real level : {}".format(
            realLabel), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
        ori_array = cv2.putText(ori_array, "Predict level : {}".format(
            preLabel), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
        ori_array_bgr = cv2.cvtColor(ori_array, cv2.COLOR_RGB2BGR)
        if (preLabel == realLabel):
            count = count + 1
            cv2.imwrite(correctPre+'/' +
                        imagePath.split(os.path.sep)[-1], ori_array_bgr)
        else:
            cv2.imwrite(wrongPre+'/' + imagePath.split(os.path.sep)
                        [-1], ori_array_bgr)
    # print("output_array==",output_array)
    # print("number correct prediction==",count)


def main():
    # Mixed_color = 0, Netting = 1, smooth = 2, Spots = 3, Wrinkled = 4

    if args["type"] == 'test':  # test model
        testVGG16()
    else:
        TransferVGGmodel()


main()

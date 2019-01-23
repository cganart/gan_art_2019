import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend
backend.set_image_dim_ordering('th')
np.random.seed(888)

# load the tinyImageNet data
X1 = np.load('X_imagenet1.npy')
X2 = np.load('X_imagenet2.npy')
X3 = np.load('X_imagenet3.npy')
X4 = np.load('X_imagenet4.npy')
y1 = np.load('y_imagenet1.npy')
y2 = np.load('y_imagenet2.npy')
y3 = np.load('y_imagenet3.npy')
y4 = np.load('y_imagenet4.npy')
# shuffle the 4 pieces of dataset and extract test sections from each
idx = np.random.choice(X1.shape[0], X1.shape[0], replace=False)
X1 = X1[idx]
y1 = y1[idx]
X1_test = X1[:750]
X1 = X1[750:]
y1_test = y1[:750]
y1 = y1[750:]
idx = np.random.choice(X2.shape[0], X2.shape[0], replace=False)
X2 = X2[idx]
y2 = y2[idx]
X2_test = X2[:750]
X2 = X2[750:]
y2_test = y2[:750]
y2 = y2[750:]
idx = np.random.choice(X3.shape[0], X3.shape[0], replace=False)
X3 = X3[idx]
y3 = y3[idx]
X3_test = X3[:750]
X3 = X3[750:]
y3_test = y3[:750]
y3 = y3[750:]
idx = np.random.choice(X4.shape[0], X4.shape[0], replace=False)
X4 = X4[idx]
y4 = y4[idx]
X4_test = X4[:750]
X4 = X4[750:]
y4_test = y4[:750]
y4 = y4[750:]

# augment the data in the training sections
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

X1 = X1.astype(np.float32)/255.0
X1 = X1.transpose((0,3,1,2))
X1_aug = []
num_augmented = 0
for X_batch, y_batch in datagen.flow(X1, y1, batch_size=25, shuffle=False):
    X1_aug.append(X_batch)
    num_augmented += 25
    if num_augmented == 40000:
        break
X1_aug = np.concatenate(X1_aug)
X1_aug = np.concatenate([X1, X1_aug])
X1 = 0
y1_new = np.concatenate([y1,y1,y1])
y1_new = y1_new[0:X1_aug.shape[0]]

X2 = X2.astype(np.float32)/255.0
X2 = X2.transpose((0,3,1,2))
X2_aug = []
num_augmented = 0
for X_batch, y_batch in datagen.flow(X2, y2, batch_size=25, shuffle=False):
    X2_aug.append(X_batch)
    num_augmented += 25
    if num_augmented == 40000:
        break
X2_aug = np.concatenate(X2_aug)
X2_aug = np.concatenate([X2, X2_aug])
X2 = 0
y2_new = np.concatenate([y2,y2,y2])
y2_new = y2_new[0:X2_aug.shape[0]]

X3 = X3.astype(np.float32)/255.0
X3 = X3.transpose((0,3,1,2))
X3_aug = []
num_augmented = 0
for X_batch, y_batch in datagen.flow(X3, y3, batch_size=25, shuffle=False):
    X3_aug.append(X_batch)
    num_augmented += 25
    if num_augmented == 40000:
        break
X3_aug = np.concatenate(X3_aug)
X3_aug = np.concatenate([X3, X3_aug])
X3 = 0
y3_new = np.concatenate([y3,y3,y3])
y3_new = y3_new[0:X3_aug.shape[0]]

X4 = X4.astype(np.float32)/255.0
X4 = X4.transpose((0,3,1,2))
X4_aug = []
num_augmented = 0
for X_batch, y_batch in datagen.flow(X4, y4, batch_size=25, shuffle=False):
    X4_aug.append(X_batch)
    num_augmented += 25
    if num_augmented == 40000:
        break
X4_aug = np.concatenate(X4_aug)
X4_aug = np.concatenate([X4, X4_aug])
X4 = 0
y4_new = np.concatenate([y4,y4,y4])
y4_new = y4_new[0:X4_aug.shape[0]]

# concatenate and shuffle the test set and the augmented training set
X_train = np.concatenate([X1_aug,X2_aug,X3_aug,X4_aug])
X1_aug = 0
X2_aug = 0
X3_aug = 0
X4_aug = 0
y_train = np.concatenate([y1_new,y2_new,y3_new,y4_new])
y_train = to_categorical(y_train)
idx = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
X_train = X_train[idx]
y_train = y_train[idx]
X_test = np.concatenate([X1_test,X2_test,X3_test,X4_test])
X_test = X_test.astype(np.float32)/255.0
X_test = X_test.transpose((0,3,1,2))
y_test = np.concatenate([y1_test,y2_test,y3_test,y4_test])
y_test = to_categorical(y_test)
idx = np.random.choice(X_test.shape[0], X_test.shape[0], replace=False)
X_test = X_test[idx]
y_test = y_test[idx]
adam = Adam(lr=0.0002, beta_1=0.5)

# define the classifier with Keras
x_input = Input(shape=(3,64,64))
x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x_input)
#x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.6)(x)
x = Dense(2048, activation='relu', kernel_initializer='he_normal')(x)
x = Dropout(0.6)(x)
x_output = Dense(200, activation='softmax', kernel_initializer='he_normal')(x)
mdl = Model(x_input, x_output)
mdl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# training
history_mdl = mdl.fit(X_train, y_train,
                      batch_size=32,
                      epochs=16,
                      verbose=2,
                      validation_data=(X_test, y_test),
                      shuffle=True)

# save model and losses
mdl.save('mdl.h5')
loss_mdl = history_mdl.history["loss"]
loss_mdl = np.array(loss_mdl)
np.savetxt("loss.txt", loss_mdl, fmt='%f')
val_loss_mdl = history_mdl.history["val_loss"]
val_loss_mdl = np.array(val_loss_mdl)
np.savetxt("val_loss.txt", val_loss_mdl, fmt='%f')
acc_mdl = history_mdl.history["acc"]
acc_mdl = np.array(acc_mdl)
np.savetxt("acc.txt", acc_mdl, fmt='%f')
val_acc_mdl = history_mdl.history["val_acc"]
val_acc_mdl = np.array(val_acc_mdl)
np.savetxt("val_acc.txt", val_acc_mdl, fmt='%f')
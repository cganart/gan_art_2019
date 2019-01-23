import numpy as np
from keras.models import load_model, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense
from keras.layers import UpSampling2D, BatchNormalization, Reshape
from keras import backend as K
K.set_image_dim_ordering('th')

# set parameters and load data
batch_size = 32
epochs = 30
# use the 64x64 version of the art dataset
X_all = np.load('X_64.npy')
num_batches = int(X_all.shape[0]/batch_size)
Losses1 = []
Losses2 = []
Losses3 = []
Losses4 = []
Losses5 = []

# load the convolutional classifier previously trained on tinyImageNet images
classifier = load_model('tinyVGG.h5')

# define gram matrix
def gram_matrix(x):
    x = K.permute_dimensions(x, (0,2,3,1))
    features = K.reshape(x, (K.shape(x)[0], -1, K.shape(x)[-1]))
    gram = K.batch_dot(features, K.permute_dimensions(features, (0,2,1)), axes=[1,2])
    return gram

# define individual loss functions used to train the AE
[C1,H1,W1] = [64,64,64]
def Loss_s1(x_true, x_hat): # style loss 1
    G_true = gram_matrix(x_true) / (C1*H1*W1)
    G_hat = gram_matrix(x_hat) / (C1*H1*W1)
    return K.sum(K.square(G_true - G_hat))

[C2,H2,W2] = [128,32,32]
def Loss_s2(x_true, x_hat): # style loss 2
    G_true = gram_matrix(x_true) / (C2*H2*W2)
    G_hat = gram_matrix(x_hat) / (C2*H2*W2)
    return K.sum(K.square(G_true - G_hat))

[C3,H3,W3] = [256,16,16]
def Loss_s3(x_true, x_hat): # style loss 3
    G_true = gram_matrix(x_true) / (C3*H3*W3)
    G_hat = gram_matrix(x_hat) / (C3*H3*W3)
    return K.sum(K.square(G_true - G_hat))

def Loss_c(x_true, x_hat): # content loss
    return K.sum(K.square(x_true - x_hat)) / (C1*H1*W1)

[C0,H0,W0] = [3,64,64]
def Loss_tv(x_true, x_hat): # total variation loss
    a = K.square(x_hat[:,:,:H0-1,:W0-1] - x_hat[:,:,1:,:W0-1]) / (C0*H0*W0)
    b = K.square(x_hat[:,:,:H0-1,:W0-1] - x_hat[:,:,:H0-1,1:]) / (C0*H0*W0)
    return K.sum(K.pow(a + b, 1.25))

# define the autoencoder with Keras
img_input = Input(shape=(3, 64, 64))
x = Conv2D(128, (3,3), padding='same')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Reshape((16*4*4,))(x)
encoded = Dense(100)(x)
x = Dense(16*4*4, activation='relu')(encoded)
x = Reshape((16,4,4))(x)
x = Conv2D(16, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(32, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(128, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(3, (3,3), padding='same')(x)
x = BatchNormalization()(x)
img_output = Activation('sigmoid')(x)
autoencoder = Model(img_input, img_output)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# define the feature extractor that uses the layers of the VGG classifier
img_input_classifier = Input(shape=(3, 64, 64))
classi = classifier.layers[1](img_input_classifier)
classi = classifier.layers[2](classi)
conv_12 = classifier.layers[3](classi)
classi = classifier.layers[4](conv_12)
classi = classifier.layers[5](classi)
classi = classifier.layers[6](classi)
classi = classifier.layers[7](classi)
conv_22 = classifier.layers[8](classi)
classi = classifier.layers[9](conv_22)
classi = classifier.layers[10](classi)
classi = classifier.layers[11](classi)
classi = classifier.layers[12](classi)
classi = classifier.layers[13](classi)
classi = classifier.layers[14](classi)
conv_33 = classifier.layers[15](classi)
style_classifier = Model(img_input_classifier, [conv_12, conv_22, conv_33])
style_classifier.compile(optimizer='adam', loss='binary_crossentropy')

# define the overall autoencoder with perceptual loss
style_classifier.trainable = False
model_input = Input(shape=(3, 64, 64))
AE_output = autoencoder(model_input)
[out_1, out_2, out_3] = style_classifier(AE_output)
model = Model(model_input, [out_1, out_2, out_3, out_1, AE_output])
# change weigths to do style loss, content loss, or a sum of the two
model.compile(optimizer='adam',
              loss=[Loss_s1, Loss_s2, Loss_s3, Loss_c, Loss_tv],
              loss_weights=[1.0, 1.0, 1.0, 0.0, 0.0])

# training
for epoch in range(epochs):

    cum_loss1 = 0.
    cum_loss2 = 0.
    cum_loss3 = 0.
    cum_loss4 = 0.
    cum_loss5 = 0.
  
    for batch_idx in range(num_batches):
        print(batch_idx)
        
        # draw a minibatch of images
        images = X_all[batch_idx*batch_size : (batch_idx+1)*batch_size].astype('float32') / 255.0
        # extract features from the minibatch; use them to train the model
        [tr_f1,tr_f2,tr_f3] = style_classifier.predict(images)
        loss_sum = model.train_on_batch(images,
                                        [tr_f1,tr_f2,tr_f3,tr_f1,images])
        cum_loss1 += loss_sum[1]
        cum_loss2 += loss_sum[2]
        cum_loss3 += loss_sum[3]
        cum_loss4 += loss_sum[4]
        cum_loss5 += loss_sum[5]
    
    Losses1.append(cum_loss1 / num_batches)
    Losses2.append(cum_loss2 / num_batches)
    Losses3.append(cum_loss3 / num_batches)
    Losses4.append(cum_loss4 / num_batches)
    Losses5.append(cum_loss5 / num_batches)

# save encoder and decoder as separate networks
np.savetxt('loss1.txt', np.array(Losses1), fmt='%f')
np.savetxt('loss2.txt', np.array(Losses2), fmt='%f')
np.savetxt('loss3.txt', np.array(Losses3), fmt='%f')
np.savetxt('loss4.txt', np.array(Losses4), fmt='%f')
np.savetxt('loss5.txt', np.array(Losses5), fmt='%f')
encoder = Model(img_input, encoded)
encoded_input = Input(shape=(100,))
deco = autoencoder.layers[19](encoded_input)
deco = autoencoder.layers[20](deco)
deco = autoencoder.layers[21](deco)
deco = autoencoder.layers[22](deco)
deco = autoencoder.layers[23](deco)
deco = autoencoder.layers[24](deco)
deco = autoencoder.layers[25](deco)
deco = autoencoder.layers[26](deco)
deco = autoencoder.layers[27](deco)
deco = autoencoder.layers[28](deco)
deco = autoencoder.layers[29](deco)
deco = autoencoder.layers[30](deco)
deco = autoencoder.layers[31](deco)
deco = autoencoder.layers[32](deco)
deco = autoencoder.layers[33](deco)
deco = autoencoder.layers[34](deco)
deco = autoencoder.layers[35](deco)
deco = autoencoder.layers[36](deco)
deco = autoencoder.layers[37](deco)
deco = autoencoder.layers[38](deco)
decoded_output = autoencoder.layers[39](deco)
decoder = Model(encoded_input, decoded_output)
encoder.save('encoder.h5')
decoder.save('decoder.h5')

# compute and save the latent codes
c = encoder.predict(X_all)
np.save("c.npy", c)
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Reshape
from keras.layers import Concatenate, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend
backend.set_image_dim_ordering('th')

# set parameters
BATCH_SIZE = 64
N_EPOCHS = 100
RANDOM_DIM = 100
adam = Adam(lr=0.0002, beta_1=0.5)
exp_replay = []
dLosses = []
gLosses = []
# load images
X_train = np.load('X.npy')
# load latent codes and normalise each dimension to mean=0 sd=1
c_train = np.load('c.npy')
CODE_DIM = c_train.shape[1]
for i in range(1,CODE_DIM):
    c_train[:,i] = (c_train[:,i] - c_train[:,i].mean())/ np.sqrt(c_train[:,i].var())
num_batches = int(X_train.shape[0]/BATCH_SIZE)
# store mean and covariance of the latent distribution as a whole
mu = np.zeros((CODE_DIM,))
Sigma = np.cov(c_train.transpose())


# define conditional generator with Keras
g_in_ran = Input(shape=(RANDOM_DIM,))
g_in_cod = Input(shape=(CODE_DIM,))
g_in = Concatenate()([g_in_ran, g_in_cod])
g = Dense(2048*4*4, activation='relu')(g_in)
g = Reshape((2048,4,4))(g)
g = Conv2DTranspose(1024, kernel_size=(4,4), strides=2, padding='same', activation='relu')(g)
g = Conv2DTranspose(512, kernel_size=(4,4), strides=2, padding='same', activation='relu')(g)
g = Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='same', activation='relu')(g)
g = Conv2DTranspose(128, kernel_size=(4,4), strides=2, padding='same', activation='relu')(g)
g_out = Conv2DTranspose(3, kernel_size=(4,4), strides=2, padding='same', activation='tanh')(g)
generator = Model([g_in_ran, g_in_cod], g_out)
generator.compile(loss='binary_crossentropy', optimizer=adam)

# define conditional discriminator with Keras
d_in_img = Input(shape=(3,128,128))
d_in_cod = Input(shape=(CODE_DIM,))
d = Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same')(d_in_img)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(256, kernel_size=(5,5), strides=(2,2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Conv2D(512, kernel_size=(5,5), strides=(2,2), padding='same')(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Flatten()(d)
d = Dense(256)(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d = Concatenate()([d, d_in_cod])
d = Dense(256)(d)
d = LeakyReLU(0.2)(d)
d = Dropout(0.3)(d)
d_out = Dense(1, activation='sigmoid')(d)
discriminator = Model([d_in_img, d_in_cod], d_out)
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# define joint CGAN network with Keras
discriminator.trainable = False
disc_condition_input = Input(shape=(CODE_DIM,))
gen_condition_input = Input(shape=(CODE_DIM,))
gan_input = Input(shape=(RANDOM_DIM,))
x = generator([gan_input, gen_condition_input])
gan_out = discriminator([x, disc_condition_input])
gan = Model([gan_input, gen_condition_input, disc_condition_input], gan_out)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# functions to sample random noise z and random code c
def generate_noise(n_samples, noise_dim):
    X = np.random.normal(0, 1, size=(n_samples, noise_dim))
    return X

def generate_random_code(n_samples):
    random_codes = np.random.multivariate_normal(mu, Sigma, n_samples)
    return random_codes

# training
for epoch in range(N_EPOCHS):

    cum_d_loss = 0.
    cum_g_loss = 0.
    # at each epoch, shuffle dataset
    idx_shuffle = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
    X_train = X_train[idx_shuffle]
    c_train = c_train[idx_shuffle]
  
    for batch_idx in range(num_batches):
        # get the set of real images and codes to be used in this iteration
        images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE].astype(np.float32)/127.5 - 1.0
        codes = c_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        
        # generate noise, random codes and then images
        noise = generate_noise(BATCH_SIZE, RANDOM_DIM)
        rand_codes = generate_random_code(BATCH_SIZE)
        generated_images = generator.predict([noise, rand_codes])
        
        # train on soft targets (add noise) and randomly flip 5% of targets
        noise_prop = 0.05
        
        # prepare labels for real data
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0,
                              high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)),
                                       size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]
        
        # train discriminator on real data
        discriminator.trainable = True
        d_loss_true = discriminator.train_on_batch([images, codes],
                                                   true_labels)
        
        # prepare labels for generated data
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0,
                             high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)),
                                       size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]
        
        # train discriminator on generated data
        d_loss_gene = discriminator.train_on_batch([generated_images,
                                                    rand_codes],
                                                   gene_labels)
        
        # store a random point for experience replay
        r_idx = np.random.randint(BATCH_SIZE)
        exp_replay.append([generated_images[r_idx], rand_codes[r_idx],
                           gene_labels[r_idx]])
        
        # if we have enough points, do experience replay
        if len(exp_replay) == BATCH_SIZE:
            generated_images = np.array([p[0] for p in exp_replay])
            codes = np.array([p[1] for p in exp_replay])
            gene_labels = np.array([p[2] for p in exp_replay])
            exp_loss = discriminator.train_on_batch([generated_images, codes],
                                                    gene_labels)
            exp_replay = []
        
        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss
        
        # train generator
        noise = generate_noise(BATCH_SIZE, RANDOM_DIM)
        random_codes = generate_random_code(BATCH_SIZE)
        discriminator.trainable = False
        g_loss = gan.train_on_batch([noise, random_codes, random_codes],
                                    np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss
    
    # store losses and periodically save models
    dLosses.append(cum_d_loss / num_batches)
    gLosses.append(cum_g_loss / num_batches)
    epoch_plus = epoch+1
    if epoch_plus % 20 == 0:
        generator.save('G_%d.h5' % epoch_plus)
    
# save losses
np.savetxt('gLoss.txt', np.array(gLosses), fmt='%f')
np.savetxt('dLoss.txt', np.array(dLosses), fmt='%f')

# generate some images and save them as a numpy array
noise = generate_noise(49, RANDOM_DIM)
random_codes = generate_random_code(49)
generated_images = generator.predict([noise, random_codes])*0.5 + 0.5
np.save('imgs.npy', generated_images)
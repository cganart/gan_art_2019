import numpy as np
from sklearn import linear_model

# load movement labels and latent codes
y = np.load("y.npy")
c = np.load("c.npy")
# normalise latent codes
for i in range(100):
    c[:,i] = (c[:,i] - c[:,i].mean())/np.sqrt(c[:,i].var())
# compute mean of each of the 20 latent movements
mu_future = np.zeros((100,))
mus = np.zeros((20,100))
mvmts = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])*1.0
for j in range(20):
    mus[j,:] = c[y==j,:].mean(axis=0)
# regress each latent means dimension on the vector of K=20 movement labels
for i in range(100):
    y_i = mus[:,i]
    x_i = mvmts.reshape((20,1))
    x_new = np.array([20.]).reshape((1,1))
    regr = linear_model.LinearRegression()
    regr.fit(x_i, y_i)
    # predict dimension i of the future latent mean
    mu_future[i] = regr.predict(x_new)
# save all latent movement means, including the future prediction
mu_all = np.concatenate([mus, mu_future.reshape((1,100))])
mv_all = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])*1.0
np.save("Desktop/mus.npy", mu_all)

n = 4
# select future mean and covariance, generate random noise and codes
noise = generate_noise(n*n, random_dim) # function from CGAN_128.py
mu_i = mu_future
Sigma_i = np.cov(z[y==19,:].transpose())
random_codes_i = np.random.multivariate_normal(mu_i, Sigma_i, n*n)
generated_images = g.predict([noise, random_codes_i])*0.5 + 0.5
plt.figure(figsize=(7,7))
for i in range(n*n):
    plt.subplot(n, n, i+1)
    #plt.imshow(generated_images[i].transpose((1,2,0)))
    plt.imshow(generated_images[i].transpose((1,2,0)))
    plt.axis('off')
plt.tight_layout(pad=-2, h_pad=-2, w_pad=-2)
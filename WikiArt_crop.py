import numpy as np
from PIL import Image
import glob

# initialize image array, names list and iteration number
X = np.empty([0,64,64,3], dtype=np.uint8)
namelist = []
i = 0
# iterate through all jpgs in the current folder
for filename in glob.glob('*.jpg'):
    try:
        # open the image and extract dimensions
        im = Image.open(filename)
        # extract the largest possible central square
        dims = im.size
        start_1 = (dims[0] - min(dims)) // 2
        start_2 = (dims[1] - min(dims)) // 2
        im = im.crop((start_1, start_2, start_1+min(dims), start_2+min(dims)))
        # reshape the central square as a 64x64 square (can change for 128x128)
        im.thumbnail((64,64), Image.ANTIALIAS)
        # turn the image into a numpy array and concatenate it to X
        image_array = np.array(im)
        image_array = image_array[np.newaxis,:,:,:]
        X = np.concatenate([X, image_array])
        namelist.append(filename)
    except:
        pass

# save the image array and names array
namelist = np.array(namelist)
np.save('X.npy', X)
np.save('namelist.npy', namelist)
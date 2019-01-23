import csv
import numpy as np
from matplotlib import pyplot as plt

# load the full image array and names array
X = np.load('X.npy')
namelist = np.load('namelist.npy')
# load the information dataset and extract information of each image
# note that the information dataset contains more rows than X
with open('all_data_info.csv', newline='', encoding="utf8") as csvfile:
    art_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    art_info = list(art_reader)
info_type = []
info_name = []
info_style = []
info_artist = []
for i in range(1,103251):
    info_type.append(art_info[i][2])
    info_name.append(art_info[i][11])
    info_style.append(art_info[i][7])
    info_artist.append(art_info[i][0])
info_type = np.array(info_type)
info_name = np.array(info_name)
info_style = np.array(info_style)
info_artist = np.array(info_artist)

# logical array that exclude weird types of paintings from the dataset
a1 = np.array(info_type!='sketch and study')
a2 = np.array(info_type!='illustration')
a3 = np.array(info_type!='design')
a4 = np.array(info_type!='interior')

# extract non-weird images matching 20 select styles (in chronological order)
a1 = np.array(info_type!='sketch and study')
a2 = np.array(info_type!='illustration')
a3 = np.array(info_type!='design')
a4 = np.array(info_type!='interior')

port = np.array(info_type=='portrait')
notport = np.array(info_type!='portrait')
land = np.array(info_type=='landscape') | np.array(info_type=='cityscape')
notland = np.array(info_type!='landscape') & np.array(info_type!='cityscape')

# extract non-weird images matching 13 select styles (in chronological order)
b = np.array(info_style=='Early Renaissance')
my_type = info_name[b & a1 & a2 & a3 & a4]
X0 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='High Renaissance')
my_type = info_name[b & a1 & a2 & a3 & a4]
X1 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Mannerism (Late Renaissance)')
my_type = info_name[b & a1 & a2 & a3 & a4]
X2 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Baroque')
my_type = info_name[b & a1 & a2 & a3 & a4]
X3 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Rococo')
my_type = info_name[b & a1 & a2 & a3 & a4]
X4 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Neoclassicism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X5 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Romanticism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X6 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Realism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X7 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Impressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X8 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Post-Impressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X9 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Fauvism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X10 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Expressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X11 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Cubism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X12 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Surrealism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X13 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Abstract Expressionism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X14 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Tachisme') | np.array(info_style=='Art Informel')
my_type = info_name[b & a1 & a2 & a3 & a4]
X15 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Lyrical Abstraction')
my_type = info_name[b & a1 & a2 & a3 & a4]
X16 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Hard Edge Painting')
my_type = info_name[b & a1 & a2 & a3 & a4]
X17 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Op Art')
my_type = info_name[b & a1 & a2 & a3 & a4]
X18 = X[np.in1d(namelist, my_type)]
b = np.array(info_style=='Minimalism')
my_type = info_name[b & a1 & a2 & a3 & a4]
X19 = X[np.in1d(namelist, my_type)]

# concatenate the 13 style subsets and create a label array
y0 = np.repeat([0],X0.shape[0])
y1 = np.repeat([1],X1.shape[0])
y2 = np.repeat([2],X2.shape[0])
y3 = np.repeat([3],X3.shape[0])
y4 = np.repeat([4],X4.shape[0])
y5 = np.repeat([5],X5.shape[0])
y6 = np.repeat([6],X6.shape[0])
y7 = np.repeat([7],X7.shape[0])
y8 = np.repeat([8],X8.shape[0])
y9 = np.repeat([9],X9.shape[0])
y10 = np.repeat([10],X10.shape[0])
y11 = np.repeat([11],X11.shape[0])
y12 = np.repeat([12],X12.shape[0])
y13 = np.repeat([13],X13.shape[0])
y14 = np.repeat([14],X14.shape[0])
y15 = np.repeat([15],X15.shape[0])
y16 = np.repeat([16],X16.shape[0])
y17 = np.repeat([17],X17.shape[0])
y18 = np.repeat([18],X18.shape[0])
y19 = np.repeat([19],X10.shape[0])
X_all = np.concatenate([X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,
                        X15,X16,X17,X18,X19])
y_all = np.concatenate([y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,
                        y15,y16,y17,y18,y19])
X_all = X_all.transpose((0,3,1,2))
# shuffle images and labels and save them as numpy files
idx_shuffle = np.random.choice(X_all.shape[0], X_all.shape[0], replace=False)
X_all_shuff = X_all[idx_shuffle]
y_all_shuff = y_all[idx_shuffle]
np.save('X.npy', X_all_shuff)
np.save('y.npy', y_all_shuff)

# plot a selection of 49 images from a particular style
figsize = (7,7)
dim = (7,7)
examples = 49
X_sample = X1[np.random.choice(X1.shape[0], examples, replace=False)]
plt.figure(figsize=figsize)
for i in range(X_sample.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(X_sample[i].transpose((1,2,0)))
    plt.axis('off')
plt.show()
### ----- CONTENT ----- ###

obtain future art prediction (linear trend only, no VAR) by following the steps:

* (1) use Classifier.py on the tinyImageNet dataset to train auxiliary classifier model.

* (2) use WikiArt_crop.py then WikiArt_select_20_movements.py to produce art dataset.
Make two versions: 64x64 and 128x128.

* (3) use AE_perceptual with style and/or content loss to obtain the latent codes.
Use the 64x64 version of the art dataset for this task.

* (4) use CGAN_128.py to train the conditional GAN using the aforementioned images and codes.

* (5) use Generate_future to make future art.
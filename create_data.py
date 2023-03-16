import tensorflow as tf
import numpy as np
import os
import random
from joblib import load, dump
from matplotlib import pyplot as plt

MEAN, STD = 120., 64.
AGE_MEAN, AGE_STD = 35.3, 15.54

tf.keras.backend.set_floatx('float32')
divs = 5

files = os.listdir('UTKFace')
# file name is [age]_[gender]_[race]_[date&time].jpg

# we set a lower and upper bound for age
files = list(filter(lambda x:6<int(x.split('_')[0])<80, files))
ages = np.array([int(x.split('_')[0]) for x in files])

random.shuffle(files)

set_size = len(files)//divs
for i in range(divs):
    
    file_set = files[i*set_size:(i+1)*set_size]
    if i == divs-1:
        file_set = files[i*set_size:]
        
    ages = np.array([int(x.split('_')[0]) for x in file_set])
    ages = (ages-AGE_MEAN)/AGE_STD 
    
    images = np.zeros((len(file_set),120,120,3), dtype='float32')

    for j, file in enumerate(file_set):
        images[j] = tf.image.resize(tf.keras.utils.img_to_array(tf.keras.utils.load_img(f"UTKFace/{file}")), (120,120))

    images = (images-MEAN)/STD
    dump(images, f'data/{i}_images.joblib')
    dump(ages, f'data/{i}_ages.joblib')
    
    plt.hist(ages, bins=30)
    plt.show()
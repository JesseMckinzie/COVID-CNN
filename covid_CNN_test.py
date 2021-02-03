from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
model = load_model('covid_final_8000_64_4x_50.h5')

my_image = plt.imread('pnuenomia.jpg')

my_image_resized = resize(my_image, (128, 128, 3))

probabilities = model.predict(np.array([my_image_resized, ]))

number_to_class = ['covid', 'normal']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[1]], "-- Probability:", probabilities[0, index[1]])
print("Second most likely class:", number_to_class[index[0]], "-- Probability:", probabilities[0, index[0]])

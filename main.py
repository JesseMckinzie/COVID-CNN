import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

"""
CATEGORIES:
0->normal
1->COVID-19
"""

batch = 50
epochs = 21  # 50

"""
Data input into data frames
"""
# Read in file names from first data set
filenames = os.listdir("/home/jmckinzi/Desktop/data")
# Categorize the data
categories = []
for filename in filenames:
    if filename.startswith('N'):
        categories.append(0)
    if filename.startswith('V'):
        categories.append(0)
    if filename.startswith('C'):
        categories.append(1)

# Build a data frame containing the image names and classification of  the firstt data set
df = pd.DataFrame({'filename': filenames,
                   'category': categories})
# Read in the second data set from a comma seperated file
df2 = pd.read_csv("/home/jmckinzi/Desktop/data_old/test/Chest_xray_Corona_Metadata.csv")  # Read file
df2.drop(['Unnamed: 0', 'Dataset_type', 'Label_1_Virus_category'], axis=1, inplace=True)  # Drop unused columns
df2['Label_2_Virus_category'].replace("COVID-19", 1, inplace=True)  # Replace strings with integers (binary classifications)
df2['Label_2_Virus_category'][df2['Label_2_Virus_category'] != 1] = 0  # Replace strings with integers (binary classifications)
df2.rename(columns={"X_ray_image_name": "filename", "Label_2_Virus_category": "category"}, inplace=True)  # Rename columns to match the first data frame
df2.drop(["Label"], axis=1, inplace=True)  # Drop the final unused data

df = pd.concat([df, df2])  # Concatenate the two data frame by stacking

df["category"] = df["category"].replace({0: 'normal', 1: 'covid'})

"""
Model creation
"""
# Create the model by creating a Sequential object
model = Sequential()

# Add layers to the model
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(2, activation='softmax'))

# Compile the model: select optimization function and loss function
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

# Split the data for training and validation
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

"""
Storing images in memory
"""
# Create a image generator and normalize the images
IDG_train = ImageDataGenerator(rescale=1./255)
# Flow images from the image generator in batches of size batch (declared early in code)
ffd_train = IDG_train.flow_from_dataframe(df_train, "/home/jmckinzi/Desktop/full_data", x_col='filename',
                                          y_col='category', target_size=(128, 128), class_mode='categorical', batch_size=batch)

# Create a image generator and normalize the images
IDG_test = ImageDataGenerator(rescale=1./255)
# Flow images from the image generator in batches of size batch (declared early in code)
ffd_test = IDG_test.flow_from_dataframe(df_test, "/home/jmckinzi/Desktop/full_data", x_col='filename', y_col='category',
                                        target_size=(128, 128), class_mode='categorical', batch_size=batch)

"""
Train the model
"""
history = model.fit_generator(ffd_train, epochs=epochs, validation_data=ffd_test, validation_steps=df_test.shape[0]//batch,
                              steps_per_epoch=df_train.shape[0]//batch)

# save the trained model to call later
model.save('covid_model.h5')

"""
Plot results
"""
# Plot training results
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Train')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

# Plot validation results
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

print('done')




# COVID-CNN
This project was completed as a part of COSC 5010: Dynamic Big Data at the University of Wyoming along with Tyler Johnson. In this project, we set out to find a method to
identify COVID-19 in chest x-rays using machine learning to expedite the process of COVID-19 testing. To do so, we chose to use a convolutional neural network (CNN), a machine learning
technique used to analyze visual imagery. To create our CNN, we used Keras, a popular Python library for CNNs that interfaces with TensorFlow. We chose this library due to its ability
to leverage the power of GPUs to reduce the run time of our algorithm. 

Once we had our model set up, we split our data of 8,815 images (2.5 Gb of data) into a train/test split of 75%/25%. After testing different layers and target functions, we ended up with a validation 
accuracy of 99.5%. For more information, see the files Presentation.pdf to see the slides presented on this poject. 

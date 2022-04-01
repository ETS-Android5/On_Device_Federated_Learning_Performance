# On Device Federated Learning Performance
This project is mainly to investigate the federated learning performance implemented on devices. The devices used in this project are Android devices, like tablets or phones. We want to figure out the memory usage, training time and power consumption of devices running federated learning. The main task of this project is to develop the App to implement federated learning on image classification tasks. The image datasets are MNIST, EMNIST and CIFAR-10.

## Hardware and Software required for this project
- three Android devices no matter it is a phone or a tablet. 
- [Android Stuidio](https://developer.android.com/studio).       This is the development environment for Android Applications.
- [Deeplearning4j](https://github.com/eclipse/deeplearning4j-examples) Framework. This is the framework for Java to implement deep learning algorithm.

## Structure of Project
There are four folders in this repo. 
- Dataset folder contains all the datasets used in this project. 
- MNIST folder contains the App sources file for MNIST dataset
- EMNIST folder contains the App sources file for EMNIST dataset
- CIFAT folder contains the App sources file for CIFAR-10 dataset

## Design of Application to Implement Federated Learning
### UI Design
<!-- ![server](/image/server.jpg) -->
<img src="/image/server.jpg" width = "30%" height = "30%" />
<img src="/image/client1.jpg" width = "30%" height = "30%" />
<img src="/image/client2.png" width = "30%" height = "30%" />

The above image shows the App User Interface. First, you should click on "Start training" of server App to generate the initial model and wait for clients connection. Then you can click on "Start training" of either clients.

### Functional Design
The first thing in functional design is to define the App permissions. This can be done in AndroidManifest.xml
- Internet Permission: use Wifi to download the dataset from Gitlab
- Storage Permission: load the dataset from internal storage and write the output back to storage
When you load the App to your devices, you also need to open the permission from the device sides to ensure it can function properly.

Then go to the main part of App design. This App functions can be divided into two parts. The first part is download the datasets using the HTTP. The second part is the federated learning implementation. 

#### Download the Dataset
All the dataset has been preprocessed to compressed file(tar.gz) and placed at the dataset folder in this repository.  In this App, the dataset will be downloaded using HTTP. There is a Java class called DataUtilities with two methods. One is to download the dataset and the other is to unzip the tar.gz files.

#### Federated Learning Implementation
Federated learning is a new machine learning technique that trains the model on distributed edge devices without sharing the data to the centralized server. The working flow of the server and clients Application is followed:

Step 1: Server generates a convolutional neural network model with zero weights and biases

Step 2: Server sends the model to all the clients by TCP socket.

Step 3: Clients receive the model from server and  train the model using its own data

Step 4: Once the training process is finished, clients send the trained model back.

Step 5: Server aggregates the weights and biases and evaluates the combined model on the test dataset.

Step 6: If the combined model reaches the target accuracy, the federated learning process stops. If not, server will send the combined model to clients to continue training.

Step 7: Repeat Step 6

##### Server Side Applications Design
The server side consists 5 classes:
- AsyncTaskRunner class：This class is intended to enable proper and easy use of the UI thread. You can get detailed infomation [here](https://developer.android.com/reference/android/os/AsyncTask).
- modelBuildAndEval class:
This class is to generate the initial model with zero weights and biases and evaluate the aggregated model performance so that it can tell clients if the training continues.
- ServiceReceive class:
This class is to do multithread of receiving files from all clients. Since there are more than one clients, multithread needs to be used.
- ServiceSend class:
This class is to do multithread of sending files from all clients. Since there are more than one clients, multithread needs to be used.
- ContinueSignalSend class:
This class is to do multithread of sending signal to all clients continuing training the model. 
- StopSignalSend class:
This class is to do multithread of sending signal to all clients continuing training the model. 

##### Client Side Applications Design
The server side consists 2 classes and 5 functions:
- AsyncTaskRunner class：This class is intended to enable proper and easy use of the UI thread. You can get detailed infomation [here](https://developer.android.com/reference/android/os/AsyncTask).
- modelTrain class:
This class is to load the training data and train the received model using its local data.
- receiveModel function:
This function is used to receive the model from server.
- sendModel function:
This function is used to send the model to server.
- receiveSignal function:
This function is used to receive the signal from server to decide if it needs to continue training the model.
- sendFile_from_client function:
This is the function to send the files using TCP socket programming.
- receiveFile function:
This is the function to receive the files using TCP socket programming.

## Performance Measurement
The main task of this project is to measure the performance of federated learning on devices in terms of training time, memory usgae and power consumption. 

### Training Time
The training time of each round is recorded in client side and the output form is csv files. For server side, it outputs the evaluation metrics in csv files.

### Memory Usage
Memory usage is observed on Android Studio embedded tools [Profiler](https://developer.android.com/studio/profile).This tool can track real-time CPU, memory, network and power usage. 

### Power Consumption
Power consumption is measured by [PowerSpy device](https://www.alciom.com/en/our-trades/products/powerspy2/).

# cnn_particlefilter  
This method proposes a new approach to automatically detect and track object using Convolutional Neural Network(CNN) and Particle Filter.  

- By using CNN as an observation model of particle filter, it is possible to simplify the observation model and improve the accuracy of the classifier.  

Learning data used to create classifier:  
 
|class|name|
|:---|:---|
|0|whole body of monkey|
|1|background|
|2|upper body of monkey|
|3|lower body of monkey|  

- In this program, since the tracking target is a Japanese macaques, the images of Japanese macaques were used for learning data.  
- Since it is necessary to recognize only the whole body monkey as a monkey, I put a partial image of the monkey.  
- Changing learning data allows you to track other animals and objects.  

# Requirements  
cnn_particlefilter requires the following to run:  

- Ubuntu16.04  
- Caffe  
- CUDA8.0  
- CUDNN6.0  
- Python2.x  
- OpenCV3.x  
# Usage  
1. Install python libraries:  
```
$ pip install scikit-image  
```  

2. Clone:  
```
$ cd caffe/examples  
$ git clone https://github.com/Rehide/cnn_particlefilter.git  
```  
3. Run:  
```
$ cd cnn_particlefilter/python  
$ python tracking.py
```  
Result Example:  
![Alt text](/python/frame.jpg)

# How to run other classsifier  
1. Prepare the data set:  
Store your own learning data in `/data/0,1,2,3...` for each class.

2. Create the LevelDB:  
```  
$ cd data  
$ python leveldb.py
```  

3. Create the classifier:  
※ *If the number of classes is changed, you need to change the num_output at line 204 of train_test.prototxt and at line 137 of cnn_particlefilter.prototxt.*  
```  
$ cd ../model  
$ ./train_full.sh 
```  

  - Then, `cnn_particlefilter_iter_60000.caffemodel` will be created.  

4. Run:  
※ *Change line 63,72,73 of tracking.py to your own model.*  
```  
$ cd ../python  
$ python tracking.py
```  

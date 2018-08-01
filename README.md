# cnn_particlefilter  
This method proposes a new approach to automatically detect and track object using Convolutional Neural Network(CNN) and Particle Filter.  
The algorithm of this method is as follows:  

1. 



Learning data used to create classifier:  
In this program, since the tracking target is a Japanese macaques, the images of Japanese macaques were used for learning data.  

|class|name|
|:---|:---|
|0|whole body of monkey|
|1|background|
|2|upper body of monkey|
|3|lower body of monkey|  

# Requirements  
cnn_particlefilter requires the following to run:  

- Ubuntu16.04  
- Caffe  
- CUDA8.0  
- CUDNN6.0  
- Python2.x  
- OpenCV3.x  
# Usage  
Install python libraries:  
`$ pip install scikit-image`  

Clone:  
```
$ cd caffe/examples  
$ git clone https://github.com/Rehide/cnn_particlefilter.git  
```  
Run:  
```
$ cd cnn_particlefilter/python`  
$ python tracking.py`
```  
Result Example:  
![Alt text](/python/frame.jpg)

# How to run other caffemodel  
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

# cnn_particlefilter  
This method proposes a new approach to automatically detect and track object using Convolutional Neural Network(CNN) and Particle Filter.  
The algorithm of this method is as follows:  

1. 



Learning data used to create classifiers:  


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
`$ cd caffe/examples`  
`$ git clone https://github.com/Rehide/cnn_particlefilter.git`  

Run:  
`$ cd cnn_particlefilter/python`  
`$ python tracking.py`  

Result Example:  
![Alt text](/python/frame.jpg)

# How to run other caffemodel  
1. Prepare the data set:  


2. Create LevelDB:  
`$ cd data`  
`$ python leveldb.py`

3. Train:  
`$ cd ../model`  
`$ ./train_full.sh`  

4. Run:  

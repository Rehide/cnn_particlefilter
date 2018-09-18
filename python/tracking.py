# -*- coding:utf-8 -*-
import numpy as np
import cv2
import csv
import math
import skimage
import sys
import time
import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()


class ParticleFilter:
    # difinition of particles
    def __init__(self):
        # number of particles
        self.SAMPLEMAX = 100
        # upper and lower limits of particle state
        self.upper_x, self.upper_y = 640, 480
        self.lower_x, self.lower_y = 0, 0
        self.upper_w, self.upper_h = 540, 380
        self.lower_w, self.lower_h = 60, 60
        # flag
        self.flag = 0

    # initialize the particles
    def initialize(self):
        self.W = np.random.random(self.SAMPLEMAX) * (self.upper_w - self.lower_w) + self.lower_w
        self.H = np.random.random(self.SAMPLEMAX) * (self.upper_h - self.lower_h) + self.lower_h
        self.X = np.random.random(self.SAMPLEMAX) * (self.upper_x - self.lower_x) + (self.lower_x - self.W/2)
        self.Y = np.random.random(self.SAMPLEMAX) * (self.upper_y - self.lower_y) + (self.lower_y - self.H/2)

    # state transition function
    def modeling(self):
        # random walk
        self.X += np.random.random(self.SAMPLEMAX) * 40 - 20
        self.Y += np.random.random(self.SAMPLEMAX) * 40 - 20
        self.W += np.random.random(self.SAMPLEMAX) * 30 - 15
        self.H += np.random.random(self.SAMPLEMAX) * 30 - 15
        # set not to exceed upper and lower limits
        for i in range(self.SAMPLEMAX):
            if (self.X[i] + self.W[i]) > self.upper_x:
                self.W[i] = self.upper_x - self.X[i]
            if (self.Y[i] + self.H[i]) > self.upper_y:
                self.H[i] = self.upper_y - self.Y[i]
            if self.X[i] > self.upper_x - self.lower_w:
                self.X[i] = self.upper_x - self.lower_w
                self.W[i] = self.lower_w - 1
            if self.Y[i] > self.upper_y - self.lower_h:
                self.Y[i] = self.upper_y - self.lower_h
                self.H[i] = self.lower_h - 1
            if self.W[i] > self.upper_w: self.W[i] = self.upper_w
            if self.H[i] > self.upper_h: self.H[i] = self.upper_h
            if self.X[i] < self.lower_x: self.X[i] = self.lower_x
            if self.Y[i] < self.lower_y: self.Y[i] = self.lower_y
            if self.W[i] < self.lower_w: self.W[i] = self.lower_w
            if self.H[i] < self.lower_h: self.H[i] = self.lower_h

    # load the caffemodel
    def caffe_preparation(self):
        mean_blob = caffe_pb2.BlobProto()
        with open('../model/mean.binaryproto') as f:
            mean_blob.ParseFromString(f.read())
        mean_array = np.asarray(
        mean_blob.data,
        dtype = np.float32).reshape(
        	(mean_blob.channels,
        	mean_blob.height,
        	mean_blob.width))
        self.classifier = caffe.Classifier(
            '../model/cnn_particlefilter.prototxt',
            '../model/cnn_particlefilter_iter_60000.caffemodel',
            mean=mean_array,
            raw_scale=255)

    # likelifood function
    def calcLikelihood(self, image):
        self.count = 0
        intensity = [] # list that stores likelifood
        # get the likelihood for each particle
        for i in range(self.SAMPLEMAX):
            y, x, w, h = self.Y[i], self.X[i], self.W[i], self.H[i]
            roi = image[math.floor(y):math.floor(y+h), math.floor(x):math.floor(x+w)]  # obtain the ROI image of particle
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) # convert the ROI image to caffe format
            img = skimage.img_as_float(img).astype(np.float32)
            predictions = self.classifier.predict([img], oversample=False) # get the probability of target object from classifier
            # count the number of times recognized as class0
            argmax_class = np.argmax(predictions)
            if argmax_class == 0: 
                self.count += 1
            intensity.append(predictions[0][int(0)]) # save the probability of the class0 as a likelihood in the list
        weights = self.normalize(intensity) # calculate the weights
        return weights

    # normalization function
    def normalize(self, predicts):
        return predicts / np.sum(predicts)

    # resampling function
    def resampling(self, weight):
        sample = []
        index = np.arange(self.SAMPLEMAX)
        # large weighted particles are stochastically chosen a lot
        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)
        return sample

    # tracking
    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        # update the particles
        self.X = self.X[index]
        self.Y = self.Y[index]
        self.W = self.W[index]
        self.H = self.H[index]
        # weighted average
        px, py, pw, ph = 0, 0, 0, 0
        px = np.average(self.X, weights = weights)
        py = np.average(self.Y, weights = weights)
        pw = np.average(self.W, weights = weights)
        ph = np.average(self.H, weights = weights)
        return px, py, pw, ph

# main
if __name__ == "__main__":
    cap = cv2.VideoCapture("movie/example.m4v")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    filter = ParticleFilter()
    filter.initialize()
    filter.caffe_preparation()

    # tracking
    while cv2.waitKey(30) < 0:
        start = time.clock() # star measuring time
        ret, frame = cap.read() # load the frame
        # x,y,w,h: coordinates of bounding box
        x, y, w, h = filter.filtering(frame)
        rx, ry = x + w, y + h
        # set not to exceed upper and lower limits
        if rx > 640: rx = 640
        if ry > 480: ry = 480
        if rx < 0: rx = 0
        if ry < 0: ry = 0
        # draw the bounding box when the number recognized as class0 exceeds the majority
        if filter.count > filter.SAMPLEMAX / 2:
            cv2.rectangle(frame, (int(x),int(y)), (int(rx),int(ry)), (0,0,255), 2)
        get_image_time = int((time.clock()-start)*1000) # finish measuring time
        cv2.putText(frame, str(1000/get_image_time) + "fps", (10,30), 2, 1, (0,255,0)) # drow the fps
        cv2.imshow("frame", frame)
        out.write(frame)
        if cv2.waitKey(30) & 0xFF == 27:break # Esc key

    # destroy resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

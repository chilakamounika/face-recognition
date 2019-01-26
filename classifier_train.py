from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from packages.classifier import training

datadir = '/media/rgukt/dc38490c-9c83-4fa4-ab97-77c0bba0f597/convocation/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow-master/pre_img'
modeldir = '/media/rgukt/dc38490c-9c83-4fa4-ab97-77c0bba0f597/convocation/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow-master/model'
classifier_filename = '/media/rgukt/dc38490c-9c83-4fa4-ab97-77c0bba0f597/convocation/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow-master/class/classifier.pkl'
print ("Training Start")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")

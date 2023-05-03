import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, features=None, w = 0.4):
        super(Model, self).__init__()
        self.w = w
        self.features = features

        #initialise weights with vgg

        #conv1
        #LRN
        #max pool

        # conv2
        # LRN
        # max pool

        # conv3
        # LRN
        # max pool

        # fc4
        # fc5
        # fc6
        # output is classification loss(box contains object or not) and 3D IoU regression loss
        # alternatively output can be confidence score





    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        score = 0
        return score

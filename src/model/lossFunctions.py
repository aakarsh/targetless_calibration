import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as rot
import numpy as np
from utils.transforms import *

def applyTransform(transform,tensor):
    ones = torch.ones(tensor.shape[1]).to('cuda:2')
    ones = torch.unsqueeze(torch.unsqueeze(ones,0),2)
    tensor = torch.cat([tensor,ones],2)
    transpose_vec = torch.transpose(tensor,1,2)
    ptCld = torch.matmul(transform.to('cuda:2'),transpose_vec.to('cuda:2'))
    ptCld = torch.transpose(ptCld,1,2)
    # remove the fourths column
    ptCld = ptCld[:,:,:3]
    return ptCld


class euclideanDistance(nn.Module):
    def __init__(self):
        super(euclideanDistance,self).__init__()

    def forward(self,predictedTransform, targetTransform, srcPtCld): 

        # COnver the data types
        targetTransform = targetTransform.float()

        # get predicted cloud
        predictedCld = applyTransform(predictedTransform, srcPtCld)

        # get target cloud 
        targetCld = applyTransform(targetTransform, srcPtCld)
        
        distance = torch.sqrt(torch.pow(targetCld[:,:,0] - predictedCld[:,:,0],2) + torch.pow(targetCld[:,:,1] - predictedCld[:,:,1],2) + torch.pow(targetCld[:,:,2] - predictedCld[:,:,2],2))
        meaneuclideandistance = torch.mean(distance)

        return(meaneuclideandistance)

class campherDistance(nn.Module):
    def __init__(self):
        super(campherDistance,self).__init__()

    def forward(self,predictedTransform, targetTransform, srcPtCld):
         # get predicted cloud
        predictedCld = applyTransform(predictedTransform, srcPtCld)

        # get target cloud 
        targetCld = applyTransform(targetTransform, srcPtCld)
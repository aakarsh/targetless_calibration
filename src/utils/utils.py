import torch
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib
import shutil
import json
import importlib
from pathlib import Path
import provider

import utils.transforms

def getTranslationRot(prediction):
    xyz = xyz = prediction[0:3,:]
    rot = prediction[3:,:].flatten()
    return(xyz, rot)

def getRotMatFromQuat(quat):
    rObj = rot.from_quat(quat)
    return(rObj.as_matrix())

def getInvTransformMat(R,T):
    # https://math.stackexchange.com/questions/152462/inverse-of-transformation-matrix
    invR = R.transpose()
    invT = np.matmul(-invR,T)
    R_T = np.vstack((np.hstack((invR,invT)),[0, 0, 0, 1.]))
    return(R_T)

def extractEulerAngles(R):
    rObj = rot.from_matrix(R)
    euler = rObj.as_euler(seq='zyx', degrees=True).reshape(3,1)
    return(euler[0],euler[1],euler[2])


def extractRotnTranslation(transformation):
    R = transformation[0:3,0:3]
    T = transformation[0:3,3]
    return(extractEulerAngles(R))


def calculateEucledianDist(pred,target,inputTensor, targetTensor):
    pred = pred.squeeze(dim=0).cpu()
    target = target.squeeze(dim=0).cpu()
    inPutCldTensor = inputTensor.transpose(2,1).cpu()
    targetCldTensor = targetTensor.cpu()

    pred = pred.data.numpy()
    target = target.data.numpy()
    inPutCld = inPutCldTensor.data.numpy()
    targetCld = targetCldTensor.data.numpy()

    '''
    Im estimating the decalibration applied.
    So th etransofrmation is inverse of the decalibration
    = decalibration ^-1 
    '''

    [predT, predQuat] = getTranslationRot(pred)
    R = getRotMatFromQuat(predQuat)
    invRt = getInvTransformMat(R, predT)

    ones = np.ones(inPutCld.shape[1]).reshape(inPutCld.shape[1],1)
    paddedinPutCld = np.hstack((inPutCld[0,:,:], ones))
    transformedptCld = np.matmul(invRt, paddedinPutCld.T).T[:,:3]


    [targetT, targetQuat] = getTranslationRot(target)
    targetR = getRotMatFromQuat(targetQuat)
    targetRT = np.vstack((np.hstack((targetR,targetT)),[0, 0, 0, 1.]))

    ones = np.ones(targetCld.shape[1]).reshape(targetCld.shape[1],1)
    paddedTargetCld = np.hstack((targetCld[0,:,:], ones))
    transformedTargetCld = np.matmul(targetRT, paddedTargetCld.T).T[:,:3]

    # calculate the eucledean distance between the the transformed and target point cloud
    eucledeanDist = np.linalg.norm(transformedTargetCld[:] - transformedptCld[:],axis=1)
  
    return(np.average(eucledeanDist), invRt, targetRT)


def exponentialMap(vec):

    u = vec[:3]
    omega = vec[3:]

    zeroTensor = torch.tensor(0.0)
   
    theta = torch.sqrt(torch.pow(omega[0],2) + torch.pow(omega[1],2) + torch.pow(omega[2],2))

    omega_cross = torch.stack([zeroTensor, -omega[2], omega[1], omega[2], zeroTensor, -omega[0], -omega[1], omega[0], zeroTensor])
    omega_cross = torch.reshape(omega_cross, [3,3])

    #Taylor's approximation for A,B and C not being used currently, approximations preferable for low values of theta

    # A = 1.0 - (tf.pow(theta,2)/factorial(3.0)) + (tf.pow(theta, 4)/factorial(5.0))
    # B = 1.0/factorial(2.0) - (tf.pow(theta,2)/factorial(4.0)) + (tf.pow(theta, 4)/factorial(6.0))
    # C = 1.0/factorial(3.0) - (tf.pow(theta,2)/factorial(5.0)) + (tf.pow(theta, 4)/factorial(7.0))

    A = torch.sin_(theta)/theta
    B = (1.0 - torch.cos_(theta))/(torch.pow(theta,2))
    C = (1.0 - A)/(torch.pow(theta,2))

    omega_cross_square = torch.matmul(omega_cross, omega_cross)

    R = torch.eye(3,3) + A*omega_cross + B*omega_cross_square

    V = torch.eye(3,3) + B*omega_cross + C*omega_cross_square
    
    Vu = torch.matmul(V,torch.unsqueeze(u,1))

    T = torch.cat([R, Vu], 1)

    # Create a 4x4 tranformation matrix
    T = torch.cat([T,torch.unsqueeze(torch.tensor([0.0, 0.0, 0.0, 1.0]),dim=0)],dim=0)
    
    T = torch.unsqueeze(T,dim=0)


    return T

def so3Transform(predictiontensor):

    # idvPred = tensor.unbind(predictiontensor)
    RT = torch.empty(0)

    for idx in torch.arange(0,predictiontensor.shape[0]):
        if idx == 0:
            RT = exponentialMap(predictiontensor[idx])
        else:
            RT_temp = exponentialMap(predictiontensor[idx])
            RT = torch.cat([RT,RT_temp],dim=0)
    print(RT)

    return RT



    
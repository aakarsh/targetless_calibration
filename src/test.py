import torch
from model.resnet import *
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib
import shutil
import json
from data_prep.dataLoader import *
import importlib
from pathlib import Path
import provider
from model import regressor
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
from model import pointcloudnet, lossFunctions
from utils.utils import *
from utils.transforms import *
from datetime import datetime
import os

from data_prep import decalibration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

cameracalibfile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/data/2011_09_26/calib_cam_to_cam.txt"


def test(model_pointnet, model_resnet, maxPool, model_regressor, loss_function ,dataLoader):
    eucledian_dist = []
    predictedRT = np.zeros((1,16), dtype = np.float32)
    targetRT = np.zeros((1,16), dtype = np.float32)


    model_resnet = model_resnet.to('cuda:1')
    model_pointnet = model_pointnet.to('cuda:1')
    maxPool = maxPool.to('cuda:2')
    model_regressor = model_regressor.to('cuda:2')
    loss_function = loss_function.to('cuda:2')

    
    # read cam projection matrix
    [P,R_02] = decalibration.readcamtocamcalibrationdata(cameracalibfile)

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):
        
        srcDepthImg, targetDepthImg, srcImage, targetImage, targetTransform, pointcloud = data

        img = torch.squeeze(srcImage,dim=0).cpu().data.numpy()
        img = (img*127.5)+127.5
        pc = torch.squeeze(pointcloud,dim=0).cpu().data.numpy()

        srcImage = srcImage.transpose(3,1).float()
        img_featuremap = model_resnet(srcImage.to('cuda:1'))

        #Add a 5x5 maxpooling 
        srcDepthImg = srcDepthImg.transpose(3,1).float()
        maxPoolImg = maxPool(srcDepthImg.to('cuda:1'))
        feature_map = model_pointnet(maxPoolImg)
        
        #img_featuremap = img_featuremap.unsqueeze(dim=2)
        aggTensor = torch.cat([feature_map,img_featuremap],dim=3)
        pred = model_regressor(aggTensor.to('cuda:2'))
        pred = torch.squeeze(pred,2)
        transformationMat = gettransformationMat(pred).to('cuda:2')
        loss = loss_function(transformationMat, targetTransform.to('cuda:2'), pointcloud.to('cuda:2'))

        # plot the points on the source image 
        
        # Send back the data
        eucledian_dist.append(loss.cpu().data.numpy())
        predictedTransform = np.expand_dims(np.ndarray.flatten(transformationMat.cpu().data.numpy()),0)
        predictedRT = np.vstack((predictedRT, predictedTransform))
        targetRT = np.vstack((targetRT, np.expand_dims(np.ndarray.flatten(targetTransform.cpu().data.numpy()),0)))


    return(eucledian_dist,predictedRT, targetRT)

def main():
    # Default parameters 
    batch_size = 1
    epochs = 50
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4

    TEST_DATASET = dataLoaderdepthimg()

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)

    #load the model
    model_common = torch.load('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/trained_model/16_12_2020_00_28_42/bestmodel_targetlesscalibration.pth')


    resnet = resnet18(pretrained=True).to('cuda:0')
    # Using the depth image for global regressor
    network_model = resnet18(pretrained=True).to('cuda:1')
    maxPool = pointcloudnet.maxPoolNet().to('cuda:1')
    regressor_model = regressor.regressor().to('cuda:1')
    loss_function = lossFunctions.euclideanDistance().to('cuda:2')

    maxPool.load_state_dict(model_common['maxPool_state_dict_2'])
    regressor_model.load_state_dict(model_common['model_state_dict_2'])

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    besteulerdistance = 100

    eulerdistances = np.empty(0)
    loss_function_vec = np.empty(0)
    predictedRT = np.zeros((1,16), dtype = np.float32)
    targetRT = np.zeros((1,16), dtype = np.float32)



    eulerDist, predictedTransformTest, targetTransformTest = test(network_model.eval(), resnet.eval(), maxPool.eval(),regressor_model.eval(), loss_function,testDataLoader)
    eulerdistances = np.append(eulerdistances,eulerDist)
    predictedRT = np.vstack((predictedRT,predictedTransformTest))
    targetRT = np.vstack((targetRT,targetTransformTest))
    print("Calculated mean Euler Distance: "+str(eulerDist))
    
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    runsummarypth = '/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/run_summaries/'+dt_string
    os.makedirs(runsummarypth)

    np.save(runsummarypth+'/eulerdistances.npy',eulerdistances)
    np.save(runsummarypth+'/predictedRT.npy',predictedRT)
    np.save(runsummarypth+'/targetRT.npy', targetRT)

    
    print("something")
    

if __name__ == "__main__":
    main()

    
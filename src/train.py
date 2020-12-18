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
from pathlib import Path
from data_prep import decalibration
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))


def test(model_pointnet, model_resnet, maxPool, model_regressor, loss_function ,dataLoader,image_save_path):
    eucledian_dist = []
    predictedRT = np.zeros((1,16), dtype = np.float32)
    targetRT = np.zeros((1,16), dtype = np.float32)


    model_resnet = model_resnet.to('cuda:1')
    model_pointnet = model_pointnet.to('cuda:1')
    maxPool = maxPool.to('cuda:2')
    model_regressor = model_regressor.to('cuda:2')
    loss_function = loss_function.to('cuda:2')
    cameracalibfile = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/data/2011_09_26/calib_cam_to_cam.txt"
    # read cam projection matrix
    [P,R_02] = decalibration.readcamtocamcalibrationdata(cameracalibfile)

    for j, data in tqdm(enumerate(dataLoader,0), total=len(dataLoader)):
        
        srcDepthImg, targetDepthImg, srcImage, targetImage, targetTransform, pointcloud = data

        img = torch.squeeze(srcImage,dim=0).cpu().data.numpy()
        img = (img*127.5)+127.5
        img = img.astype(np.uint8)
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

        # write image transforms
        rot_ = torch.squeeze(transformationMat,dim=0).cpu().data.numpy()
        [points,color] = decalibration.projpointcloud2imgplane(pc,(-24.9,2),(-45,45), rot_[:3,:3], rot_[:3,3].reshape(3,1), P)
        ptsnimg = decalibration.displayprojectedptsonimg(points,color,img)
        cv2.imwrite(image_save_path+'transformed_img_'+str(j)+'.jpg',ptsnimg)
        cv2.imwrite(image_save_path+'original_img'+str(j)+'.jpg',img)
        
        # Send back the data
        eucledian_dist.append(loss.cpu().data.numpy())
        predictedTransform = np.expand_dims(np.ndarray.flatten(transformationMat.cpu().data.numpy()),0)
        predictedRT = np.vstack((predictedRT, predictedTransform))
        targetRT = np.vstack((targetRT, np.expand_dims(np.ndarray.flatten(targetTransform.cpu().data.numpy()),0)))


    return(eucledian_dist,predictedRT, targetRT)


def main():

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    savedir = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/trained_model/"+dt_string
    trainingmetadat = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/metadata/"+dt_string
    os.makedirs(savedir)

    # Default parameters 
    batch_size = 1
    epochs = 50
    learning_rate = 0.0001 # 10^-5
    decay_rate = 1e-4

    # Hyper Parameters 
    
    totalsize = dataLoaderdepthimg().dataset.shape[0]

    # USe 30 % of the data as testing dataset
    cutoff = int(np.floor(totalsize*0.7))
    TRAIN_DATASET = dataLoaderdepthimg()
    TEST_DATASET = dataLoaderdepthimg()

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=False, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)
    #MODEL = importlib.import_module(pointcloudnet)

    # empty the CUDA memory
    torch.cuda.empty_cache()

    #network_model = pointcloudnet.pointcloudnet(layers=[1, 1, 1, 1, 1, 1])
    resnet = resnet18(pretrained=True).to('cuda:0')
    # Using the depth image for global regressor
    network_model = resnet18(pretrained=True).to('cuda:1')
    maxPool = pointcloudnet.maxPoolNet().to('cuda:1')
    regressor_model = regressor.regressor().to('cuda:1')
    loss_function = lossFunctions.euclideanDistance().to('cuda:2')

    try:
        model_common = torch.load('/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/trained_model/16_12_2020_00_28_42/bestmodel_targetlesscalibration.pth')
        maxPool.load_state_dict(model_common['maxPool_state_dict_2'])
        regressor_model.load_state_dict(model_common['model_state_dict_2'])
    except:
        print("error loading module")

    

    optimizer = torch.optim.Adam(
        network_model.parameters(),
        lr = learning_rate,
        betas = (0.9, 0.999),
        eps = 1e-08,
        weight_decay = decay_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    global_epoch = 0
    global_step = 0
    besteulerdistance = 100


    eulerdistances = np.empty(0)
    loss_function_vec = np.empty(0)
    predictedRT = np.zeros((1,16), dtype = np.float32)
    targetRT = np.zeros((1,16), dtype = np.float32)


    # Training 
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        for batch_no, data in tqdm(enumerate(trainDataLoader,0), total=len(trainDataLoader), smoothing=0.9):
            srcDepthImg, targetDepthImg, srcImage, targetImage, targetTransform, pointcloud = data 
            
            optimizer.zero_grad()

            torch.autograd.set_detect_anomaly(True)

            '''
            Since using a  pretrained model Put themodel to evaluation mode 
            we do not want to train the model 
            '''

            resnet = resnet.to('cuda:0')
            network_model = network_model.to('cuda:1')
            maxPool = maxPool.to('cuda:1')
            regressor_model = regressor_model.to('cuda:1')
            loss_function = loss_function.to('cuda:2')


            network_model = network_model.eval()
            resnet = resnet.eval()
            
            maxPool = maxPool.train()

            srcImage = srcImage.transpose(3,1).float().to('cuda:0')
            img_featuremap = resnet(srcImage)

            #Add a 5x5 maxpooling 
            srcDepthImg = srcDepthImg.transpose(3,1).float().to('cuda:1')
            maxPoolImg = maxPool(srcDepthImg)
            feature_map = network_model(maxPoolImg)
            
            #img_featuremap = img_featuremap.unsqueeze(dim=2)
            aggTensor = torch.cat([feature_map,img_featuremap.to('cuda:1')],dim=3)
            pred = regressor_model(aggTensor)

            # Get the the transformation matrix
            pred = torch.squeeze(pred,2).to('cuda:2')
            # transformationMat = so3Transform(pred)
            transformationMat = gettransformationMat(pred)
            
#            print(transformationMat)

            loss = loss_function(transformationMat, targetTransform.to('cuda:2'), pointcloud.to('cuda:2'))
            loss_function_vec = np.append(loss_function_vec,loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            global_step += 1

        with torch.no_grad():

            epoch_save_pth = trainingmetadat+'/epoch'+str(global_epoch)+'/'
            img_save_pth = epoch_save_pth+'transformed_imgs/'
            os.makedirs(img_save_pth)
            summaries_save_path = epoch_save_pth+'runsummaries'
            os.makedirs(summaries_save_path)

            eulerDist, predictedTransformTest, targetTransformTest = test(network_model.eval(), resnet.eval(), maxPool.eval(),regressor_model.eval(), loss_function,testDataLoader,img_save_pth)
    
            
            np.save(summaries_save_path+'/eulerdistances.npy',eulerDist)
            np.save(summaries_save_path+'/predictedRT.npy',predictedTransformTest)
            np.save(summaries_save_path+'/targetRT.npy', targetTransformTest)
            print("Calculated mean Euler Distance: "+str(np.mean(eulerDist))+" and the loss: "+str(loss_function_vec[global_epoch])+" for Global Epoch: "+str(global_epoch))

            eulerDist = np.mean(eulerDist)

            eulerdistances = np.append(eulerdistances,eulerDist)
            predictedRT = np.vstack((predictedRT,predictedTransformTest))
            targetRT = np.vstack((targetRT,targetTransformTest))

            if(eulerDist<besteulerdistance):
                besteulerdistance = eulerDist
                # make sure you save the model as checkpoint
                print("saving the model")
                savepath = savedir+"/bestmodel_targetlesscalibration.pth"

                state = {
                    'epoch': global_epoch,
                    'bestEulerDist': besteulerdistance,
                    'maxPool_state_dict_2':maxPool.state_dict(),
                    'model_state_dict_2':regressor_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state,savepath)
            
        global_epoch += 1
    
    runsummarypth = '/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/run_summaries/'+dt_string
    os.makedirs(runsummarypth)

    np.save(runsummarypth+'/eulerdistances.npy',eulerdistances)
    np.save(runsummarypth+'/predictedRT.npy',predictedRT)
    np.save(runsummarypth+'/targetRT.npy', targetRT)

    
    print("something")
        
            

if __name__ == "__main__":
    main()



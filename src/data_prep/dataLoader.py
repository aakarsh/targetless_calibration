import shutil
import json  
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as rot
import cv2
import math
import csv
import imageio as io



def normalizePtCld(pointcloud):
    centroid = np.mean(pointcloud, axis=0)
    pointcloud = pointcloud - centroid
    m = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
    pointcloud = pointcloud / m
    return pointcloud

def readvelodynepointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float32).reshape(-1, 4)
    intensity_data = pointcloud[:,3]
    
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3], intensity_data)

def readtransformedpointcloud(path_to_file):
    '''
    The velodyne data that is presented is in the form of a np array written as binary data.
    So to read the file, we use the inbuilt fuction form the np array to rad from the file
    '''

    pointcloud = np.fromfile(path_to_file, dtype=np.float64).reshape(-1,3)
    # Return points ignoring the reflectivity 
    return(pointcloud[:,:3])

def normalizeImg(image):
    # Convert the image into float
    image = image.astype('float32')
    image = np.divide(image,255)
    return (image)

def preprocessInputImg(image):
    scale = 224/image.shape[0]
    newWidth = math.floor(image.shape[1]*scale)
    newHeight = math.floor(image.shape[0]*scale)
    resized_img = cv2.resize(image,(newWidth,newHeight))
    normalized_img = normalizeImg(resized_img)
    return(normalized_img)


class dataLoader(Dataset):

    def __init__ (self, filename='/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/targetless_calibration/data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/agumenteddata/angles_summary.json'):
        self.datafile = filename
        file_descriptor = open(self.datafile,'r')
        self.data = json.load(file_descriptor)

    def __len__(self):
        return(len(self.data))

    def __getitem__(self, key):
        return(self.getItem(str(key)))

    def getItem(self, key):
        # read the point cloud 
        ptCldFileName = self.data[key]["point_filename"]
        targetCldFileName = self.data[key]["target_filename"]
        imageFileName = self.data[key]["image_filename"]
        inputImg = cv2.imread(imageFileName)
        try:
            ptCld = readtransformedpointcloud(ptCldFileName)
        except:
            ptCld, intensity = readvelodynepointcloud(ptCldFileName)
        
        targetCld, intensity = readvelodynepointcloud(targetCldFileName)

        if(ptCld.shape[0] != targetCld.shape[0]):
#            print("The size of the input cloud and the target cloud is not the same")
#            print("Input Point Cloud Size = "+str(ptCld.shape[0])+"\tTransformed Point Cloud Size = "+str(targetCld.shape[0]))
            ptCld = targetCld
            

        transform = self.data[key]["transform"].strip("[[]]").split(" ")
        R_t = []
        for i in range(len(transform)):
            try:
                R_t.append(float(transform[i]))
            except ValueError:
                pass
        transform = np.array(R_t, dtype=float).reshape(7,1)

        # Normalize the image
        processedImg = preprocessInputImg(inputImg)

        '''
        translation = transform[:3]
        rotation = transform[4:]
        '''
        # Pending normalization of point cloud
        # ptCld[:,0:3] = normalizePtCld(ptCld[:,0:3])
        return (ptCld, processedImg, transform, targetCld)

class dataLoaderdepthimg(Dataset):
    def __init__ (self, filename='/home/akshay/calibnet/parsed_set.txt'):
        
        datafile = filename
        self.dataset = np.loadtxt(filename, dtype = str)
        self.data = {}

        self.transformed_depth_map = np.ndarray.tolist(self.dataset[:,0])
        self.original_depth_map = np.ndarray.tolist(self.dataset[:,1])
        self.image_02 = np.ndarray.tolist(self.dataset[:,2])
        self.image_03 = np.ndarray.tolist(self.dataset[:,3])
        self.transforms = self.dataset[:,4:20].astype(float)
        self.pointcloud = np.ndarray.tolist(self.dataset[:,20])
        

        
    def __len__(self):
        return(len(self.dataset))

    def __getitem__(self, key):
        return(self.getItem(str(key)))

    def getItem(self, key):
        srcdepthimgfp = self.transformed_depth_map[int(key)] 
        targetdepthimgfp = self.original_depth_map[int(key)] 
        colorimagefp = self.image_02[int(key)] 
        colorimage_2fp = self.image_03[int(key)]
        tragettransform = self.transforms[int(key)].reshape(4,4)
        ptcldfilename = self.pointcloud[int(key)]
        
        '''
        Read the taransformed depth images
        '''
        srcdepthimg = cv2.imread(srcdepthimgfp).astype(float)
        [imgWD,imgHT,channel] = srcdepthimg.shape
        srcdepthimg[0:5,:,:] = 0.0
        srcdepthimg[:,0:5] = 0.0
        srcdepthimg[imgWD-5:,:] = 0.0
        srcdepthimg[:,imgHT-5:] = 0.0
        # Normalize the image
        srcdepthimg = (srcdepthimg - 40.0)/40.0
        '''
        Read the target depth image
        '''
        targetdepthimg = cv2.imread(targetdepthimgfp)
        [imgWD,imgHT,channel] = targetdepthimg.shape
        targetdepthimg[0:5,:,:] = 0.0
        targetdepthimg[:,0:5] = 0.0
        targetdepthimg[imgWD-5:,:] = 0.0
        targetdepthimg[:,imgHT-5:] = 0.0
        # Normalize the image
        targetdepthimg = (targetdepthimg - 40.0)/40.0

        '''
        Read the color image image_02 
        '''
        colorimage = cv2.imread(colorimagefp)
        [imgWD,imgHT,channel] = colorimage.shape
        colorimage[0:5,:,:] = 0.0
        colorimage[:,0:5] = 0.0
        colorimage[imgWD-5:,:] = 0.0
        colorimage[:,imgHT-5:] = 0.0
        # Normalize the image
        colorimage = (colorimage - 127.5)/127.5

        '''
        Read the color image image_03
        '''
        colorimage_2 = cv2.imread(colorimage_2fp)
        [imgWD,imgHT,channel] = colorimage_2.shape
        colorimage_2[0:5,:,:] = 0.0
        colorimage_2[:,0:5] = 0.0
        colorimage_2[imgWD-5:,:] = 0.0
        colorimage_2[:,imgHT-5:] = 0.0
        # Normalize the image
        colorimage_2 = (colorimage_2 - 127.5)/127.5

        '''
        Read the source point cloud
        '''
        src_ptcld, intensity = readvelodynepointcloud(ptcldfilename)
        
        #subsample poitclouds



        return([srcdepthimg, targetdepthimg, colorimage, colorimage_2, tragettransform, src_ptcld])



if __name__ == "__main__":
    obj = dataLoader()
    x0, x1, x2, x3 = obj.__getitem__('15')
    print("Tested")
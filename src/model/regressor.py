import torch
import torch.nn as nn



class regressor(nn.Module):
    def __init__(self):
        super(regressor,self).__init__()

        # The input to the feature matching network will be [24 2 512]
        # Setting the kernel size to 1
        '''
        I/P Tensor size = [5 512 39 24]
        Needed Stride = [2,2]
        '''
        channels = 512

        self.conv2 = nn.Conv2d(512,512,kernel_size=[3,3],stride=[2,2], padding=5)
        self.conv2x2 = nn.Conv2d(512,512,kernel_size=[1,2],stride=[2,2], padding=5)
        self.batchnormalization = nn.BatchNorm2d(512)
        self.Relu = nn.ReLU(inplace=True)

        self.convasFCx = nn.Conv1d(56320,3,kernel_size=1,stride=1)
        self.convasFCq = nn.Conv1d(56320,4,kernel_size=1,stride=1)  

        self.dropout = nn.Dropout2d()

        '''
        channels = 2
        self.conv1x1B0 = nn.Conv1d(2, 64,1)
        nn.init.xavier_uniform_(self.conv1x1B0.weight)
        self.conv1x1B1 = nn.Conv1d(64, 128,1)
        nn.init.xavier_uniform_(self.conv1x1B1.weight)
        self.conv1x1B2 = nn.Conv1d(128, 256,1)
        nn.init.xavier_uniform_(self.conv1x1B2.weight)

        self.FC0 = nn.Linear(256,128)
        nn.init.xavier_uniform_(self.FC0.weight)
        self.FC1 = nn.Linear(128,64)
        nn.init.xavier_uniform_(self.FC1.weight)
        self.FC2 = nn.Linear(64,7)
        nn.init.xavier_uniform_(self.FC2.weight)
        self.bn0 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.lRelu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.avgpool_regress = nn.AdaptiveAvgPool1d(1)
        '''


    def forward(self, x):

        # [5 512 39 24]
        x = self.conv2(x)
        x = self.batchnormalization(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.batchnormalization(x)
        x = self.Relu(x)
        x = self.conv2x2(x)
        x = self.batchnormalization(x)
        x = self.Relu(x)

        tr = self.conv2(x)
        tr = self.batchnormalization(tr)
        tr = self.Relu(tr)
        tr = self.dropout(tr)

        tr = tr.reshape(tr.shape[0],tr.shape[1]*tr.shape[2]*tr.shape[3],1)
        tr = self.convasFCx(tr)

        rot = self.conv2(x)
        rot = self.batchnormalization(rot)
        rot = self.Relu(rot)
        rot = self.dropout(rot)

        rot = rot.reshape(tr.shape[0],rot.shape[1]*rot.shape[2]*rot.shape[3],1)
        rot = self.convasFCq(rot)




        """
        Feature matching
        """
        '''
        x = self.conv1x1B0(x)
        x = self.lRelu(self.bn0(x))
        x = self.conv1x1B1(x)
        x = self.lRelu(self.bn1(x))
        x = self.conv1x1B2(x)
        x = self.lRelu(self.bn2(x))
        x = self.avgpool(x)
        '''

        """
        Regression
        """
        '''
        x = self.FC0(x)
        x = self.FC1(x)
        x = self.FC2(x)

        x = self.avgpool_regress(x.transpose(2,1))
        '''

        return(torch.cat([tr,rot],dim=1))





         

         
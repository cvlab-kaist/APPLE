import os
import torch
import pickle
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
try:
    from libs.face import FaceLandmarksDetector
except:
    from face import FaceLandmarksDetector

class IrisDetector():
    def __init__(self) -> None:    
        self.iris_detector = MediaPipeIris(pretrained=True)
        self.face_landmarks_detector = FaceLandmarksDetector()
        self.patch_size = 64

        self.tfm = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)), # depends on your model input size
        ])
        self.horizontal_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])

    def preprocess(self, image, face_landmarks_detection):
        key_leye = ["leftEyeUpper0", "leftEyeLower0"]
        key_reye = ["rightEyeUpper0", "rightEyeLower0"]

        left_eye_indices = self.face_landmarks_detector.get_face_landmarks_indices_by_regions(key_leye)
        right_eye_indices = self.face_landmarks_detector.get_face_landmarks_indices_by_regions(key_reye)
        
        left_eye_region = face_landmarks_detection[left_eye_indices, :]
        right_eye_region = face_landmarks_detection[right_eye_indices, :]

        left_eye_width = left_eye_region[np.argmax(left_eye_region[:, 0]), 0] - left_eye_region[np.argmin(left_eye_region[:, 0]), 0]
        right_eye_width = right_eye_region[np.argmax(right_eye_region[:, 0]), 0] - right_eye_region[np.argmin(right_eye_region[:, 0]), 0]
        
        left_eye_propCrop_size = int(left_eye_width * 4/2)
        right_eye_propCrop_size = int(right_eye_width * 4/2)


        left_eye_center = np.mean(left_eye_region, axis=0).astype(np.int32)
        right_eye_center = np.mean(right_eye_region, axis=0).astype(np.int32)
        
        left_eye_image = image[left_eye_center[1]-left_eye_propCrop_size//2:left_eye_center[1]+left_eye_propCrop_size//2, left_eye_center[0]-left_eye_propCrop_size//2:left_eye_center[0]+left_eye_propCrop_size//2]
        right_eye_image = image[right_eye_center[1]-right_eye_propCrop_size//2:right_eye_center[1]+right_eye_propCrop_size//2, right_eye_center[0]-right_eye_propCrop_size//2:right_eye_center[0]+right_eye_propCrop_size//2]

        scaler_left = left_eye_image.shape[1]/self.patch_size
        scaler_right = right_eye_image.shape[1]/self.patch_size
        left_eye_image = np.array(self.tfm(Image.fromarray(left_eye_image)))
        right_eye_image = np.array(self.tfm(Image.fromarray(right_eye_image)))

        return left_eye_image, right_eye_image, (left_eye_center, scaler_left), (right_eye_center, scaler_right) 

    def postprocess(self, eye_contour, eye_iris, config):
        eye_center, scaler = config
        _eye_center = self.patch_size*scaler//2
        offset = eye_center-_eye_center
        eye_contour = np.ones_like(eye_contour)*offset + eye_contour*scaler
        eye_iris = np.ones_like(eye_iris)*offset + eye_iris*scaler

        return eye_contour, eye_iris

    def predict(self, image, isLeft=True):
        self.iris_detector.eval()
        
        if isLeft is False:
            image = np.array(self.horizontal_flip(Image.fromarray(image)))

        t_image = torch.from_numpy(np.transpose(image, (2, 0, 1))/255.0).unsqueeze(0)
        t_image = Variable(t_image).type(torch.float)
        eyeContour, iris = self.iris_detector(t_image.to(next(self.iris_detector.parameters()).device))

        eyeContour = eyeContour.squeeze().reshape(-1, 3).detach().cpu().numpy()
        iris = iris.squeeze().reshape(-1, 3).detach().cpu().numpy()

        if isLeft is False:
            eyeContour[:, 0] = np.ones_like(eyeContour[:, 0])*64-eyeContour[:, 0]
            iris[:, 0] = np.ones_like(iris[:, 0])*64-iris[:, 0]
        return eyeContour, iris

class Conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32) -> None:
        super(Conv2d_block, self).__init__() 
        self.Conv2D_0 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.Prelu = nn.PReLU(num_parameters=hidden_channels) 
        self.DepthwiseConv2d = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1, groups=hidden_channels) 
        self.Conv2D_1 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
    
    def load(self, Conv2D_0_weight=None, Conv2D_0_bias=None, Prelu_weight=None, 
            DepthwiseConv2d_weight=None, DepthwiseConv2d_bias=None,
            Conv2D_1_weight=None, Conv2D_1_bias=None):
        if Conv2D_0_weight is not None:
            self.Conv2D_0.weight = torch.nn.Parameter(torch.from_numpy(Conv2D_0_weight))
        if Conv2D_0_bias is not None:
            self.Conv2D_0.bias = torch.nn.Parameter(torch.from_numpy(Conv2D_0_bias))
        if Prelu_weight is not None:
            self.Prelu.weight = torch.nn.Parameter(torch.from_numpy(Prelu_weight))
        if DepthwiseConv2d_weight is not None:
            self.DepthwiseConv2d.weight = torch.nn.Parameter(torch.from_numpy(DepthwiseConv2d_weight))
        if DepthwiseConv2d_bias is not None:
            self.DepthwiseConv2d.bias = torch.nn.Parameter(torch.from_numpy(DepthwiseConv2d_bias))
        if Conv2D_1_weight is not None:
            self.Conv2D_1.weight = torch.nn.Parameter(torch.from_numpy(Conv2D_1_weight))
        if Conv2D_1_bias is not None:
            self.Conv2D_1.bias = torch.nn.Parameter(torch.from_numpy(Conv2D_1_bias))
    
    def forward(self, x):
        x = self.Conv2D_0(x)
        x = self.Prelu(x)
        x = self.DepthwiseConv2d(x)
        x = self.Conv2D_1(x)
        return x

class Conv2d_block_v2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, pad=True) -> None:
        super(Conv2d_block_v2, self).__init__() 
        self.pad = pad
        self.Conv2D_0 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=2, stride=2)
        self.Prelu = nn.PReLU(num_parameters=hidden_channels) 
        self.DepthwiseConv2d = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1, groups=hidden_channels) 
        self.Conv2D_1 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        self.Maxpool2D = nn.MaxPool2d(kernel_size=2, stride=2)

    def load(self, Conv2D_0_weight=None, Conv2D_0_bias=None, Prelu_weight=None, 
            DepthwiseConv2d_weight=None, DepthwiseConv2d_bias=None,
            Conv2D_1_weight=None, Conv2D_1_bias=None):
        if Conv2D_0_weight is not None:
            self.Conv2D_0.weight = torch.nn.Parameter(torch.from_numpy(Conv2D_0_weight))
        if Conv2D_0_bias is not None:
            self.Conv2D_0.bias = torch.nn.Parameter(torch.from_numpy(Conv2D_0_bias))
        if Prelu_weight is not None:
            self.Prelu.weight = torch.nn.Parameter(torch.from_numpy(Prelu_weight))
        if DepthwiseConv2d_weight is not None:
            self.DepthwiseConv2d.weight = torch.nn.Parameter(torch.from_numpy(DepthwiseConv2d_weight))
        if DepthwiseConv2d_bias is not None:
            self.DepthwiseConv2d.bias = torch.nn.Parameter(torch.from_numpy(DepthwiseConv2d_bias))
        if Conv2D_1_weight is not None:
            self.Conv2D_1.weight = torch.nn.Parameter(torch.from_numpy(Conv2D_1_weight))
        if Conv2D_1_bias is not None:
            self.Conv2D_1.bias = torch.nn.Parameter(torch.from_numpy(Conv2D_1_bias))
    
    def forward(self, x, x1):
        x = self.Conv2D_0(x)
        x = self.Prelu(x)
        x = self.DepthwiseConv2d(x)
        x = self.Conv2D_1(x)

        x1 = self.Maxpool2D(x1)

        if self.pad is True:
            x1 = torch.cat([x1, torch.zeros_like(x1)] , dim=1)
        return x + x1


class MediaPipeIris(nn.Module):
    def __init__(self, pretrained=False, ckpt_path="./data/iris_landmark.pth", weights_path=None) -> None:
        super(MediaPipeIris, self).__init__()

        self.padding = nn.ZeroPad2d((0,2,0,2))
        self.Conv2D_0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2)

        self.Prelu_0 = nn.PReLU(num_parameters=64)
        self.Conv2D_block_0 = Conv2d_block(in_channels=64,out_channels=64, hidden_channels=32)

        self.Prelu_1 = nn.PReLU(num_parameters=64)
        self.Conv2D_block_1 = Conv2d_block(in_channels=64,out_channels=64, hidden_channels=32)

        self.Prelu_2 = nn.PReLU(num_parameters=64)
        self.Conv2D_block_2 = Conv2d_block(in_channels=64,out_channels=64, hidden_channels=32)

        self.Prelu_3 = nn.PReLU(num_parameters=64)
        self.Conv2D_block_3 = Conv2d_block(in_channels=64,out_channels=64, hidden_channels=32)

        self.Prelu_4 = nn.PReLU(num_parameters=64)
        self.Conv2D_block_v2_0 = Conv2d_block_v2(in_channels=64, out_channels=128, hidden_channels=64)

        self.Prelu_5 = nn.PReLU(num_parameters=128)
        self.Conv2D_block_4 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.Prelu_6 = nn.PReLU(num_parameters=128)
        self.Conv2D_block_5 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.Prelu_7 = nn.PReLU(num_parameters=128)
        self.Conv2D_block_6 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.Prelu_8 = nn.PReLU(num_parameters=128)
        self.Conv2D_block_7 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.Prelu_9 = nn.PReLU(num_parameters=128)
        self.Conv2D_block_v2_1 = Conv2d_block_v2(in_channels=128, out_channels=128, hidden_channels=64, pad=False)

        self.Prelu_10 = nn.PReLU(num_parameters=128)


        # eye contour
        self.eyeContour_Conv2D_block_0 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)
        self.eyeContour_Prelu_1 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_1 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.eyeContour_Prelu_2 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_v2_0 = Conv2d_block_v2(in_channels=128,out_channels=128, hidden_channels=64, pad=False)

        self.eyeContour_Prelu_3 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_2 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.eyeContour_Prelu_4 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_3 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.eyeContour_Prelu_5 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_v2_1 = Conv2d_block_v2(in_channels=128,out_channels=128, hidden_channels=64, pad=False)

        self.eyeContour_Prelu_6 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_4 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.eyeContour_Prelu_7 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_block_5 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.eyeContour_Prelu_8 = nn.PReLU(num_parameters=128)
        self.eyeContour_Conv2D_out = nn.Conv2d(in_channels=128,out_channels=213, kernel_size=2)

        # eye iris
        self.iris_Conv2D_block_0 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_1 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_1 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_2 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_v2_0 = Conv2d_block_v2(in_channels=128,out_channels=128, hidden_channels=64, pad=False)

        self.iris_Prelu_3 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_2 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_4 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_3 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_5 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_v2_1 = Conv2d_block_v2(in_channels=128,out_channels=128, hidden_channels=64, pad=False)

        self.iris_Prelu_6 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_4 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_7 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_block_5 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)

        self.iris_Prelu_8 = nn.PReLU(num_parameters=128)
        self.iris_Conv2D_out = nn.Conv2d(in_channels=128,out_channels=15, kernel_size=2)


        if pretrained: 
            if weights_path is not None or not os.path.exists(ckpt_path):
                if weights_path is None:
                    weights_path = './data/weights.pkl'
                with open(weights_path, 'rb') as picklefile:
                    d = pickle.load(picklefile)
                self.Conv2D_0.weight = torch.nn.Parameter(torch.from_numpy(d['Conv2D_0_weight']))
                self.Conv2D_0.bias = torch.nn.Parameter(torch.from_numpy(d['Conv2D_0_bias']))
                self.Prelu_0.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_0_weight']))
                self.Conv2D_block_0.load(Conv2D_0_weight=d['Conv2D_block_0_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_0_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_0_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_0_Conv2D_1_bias'])
                self.Prelu_1.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_1_weight']))
                self.Conv2D_block_1.load(Conv2D_0_weight=d['Conv2D_block_1_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_1_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_1_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_1_Conv2D_1_bias'])
                self.Prelu_2.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_2_weight']))
                self.Conv2D_block_2.load(Conv2D_0_weight=d['Conv2D_block_2_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_2_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_2_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_2_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_2_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_2_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_2_Conv2D_1_bias'])
                self.Prelu_3.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_3_weight']))
                self.Conv2D_block_3.load(Conv2D_0_weight=d['Conv2D_block_3_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_3_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_3_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_3_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_3_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_3_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_3_Conv2D_1_bias'])
                self.Prelu_4.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_4_weight']))
                self.Conv2D_block_v2_0.load(Conv2D_0_weight=d['Conv2D_block_v2_0_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_v2_0_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_v2_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_v2_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_v2_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_v2_0_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_v2_0_Conv2D_1_bias'])
                self.Prelu_5.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_5_weight']))
                self.Conv2D_block_4.load(Conv2D_0_weight=d['Conv2D_block_4_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_4_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_4_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_4_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_4_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_4_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_4_Conv2D_1_bias'])
                self.Prelu_6.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_6_weight']))

                self.Conv2D_block_5.load(Conv2D_0_weight=d['Conv2D_block_5_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_5_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_5_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_5_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_5_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_5_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_5_Conv2D_1_bias'])

                self.Prelu_7.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_7_weight']))
                self.Conv2D_block_6.load(Conv2D_0_weight=d['Conv2D_block_6_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_6_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_6_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_6_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_6_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_6_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_6_Conv2D_1_bias'])

                self.Prelu_8.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_8_weight']))
                self.Conv2D_block_7.load(Conv2D_0_weight=d['Conv2D_block_7_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_7_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_7_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_7_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_7_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_7_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_7_Conv2D_1_bias'])

                self.Prelu_9.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_9_weight']))
                self.Conv2D_block_v2_1.load(Conv2D_0_weight=d['Conv2D_block_v2_1_Conv2D_0_weight'],Conv2D_0_bias=d['Conv2D_block_v2_1_Conv2D_0_bias'], Prelu_weight=d['Conv2D_block_v2_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["Conv2D_block_v2_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["Conv2D_block_v2_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['Conv2D_block_v2_1_Conv2D_1_weight'],Conv2D_1_bias=d['Conv2D_block_v2_1_Conv2D_1_bias'])

                self.Prelu_10.weight = torch.nn.Parameter(torch.from_numpy(d['Prelu_10_weight']))

                self.eyeContour_Conv2D_block_0.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_0_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_0_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_0_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_0_Conv2D_1_bias'])

                self.eyeContour_Prelu_1.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_1_weight']))
                self.eyeContour_Conv2D_block_1.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_1_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_1_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_1_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_1_Conv2D_1_bias'])

                self.eyeContour_Prelu_2.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_2_weight']))
                self.eyeContour_Conv2D_block_v2_0.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_v2_0_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_v2_0_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_v2_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_v2_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_v2_0_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_v2_0_Conv2D_1_bias'])

                self.eyeContour_Prelu_3.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_3_weight']))
                self.eyeContour_Conv2D_block_2.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_2_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_2_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_2_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_2_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_2_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_2_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_2_Conv2D_1_bias'])

                self.eyeContour_Prelu_4.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_4_weight']))
                self.eyeContour_Conv2D_block_3.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_3_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_3_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_3_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_3_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_3_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_3_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_3_Conv2D_1_bias'])

                self.eyeContour_Prelu_5.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_5_weight']))
                self.eyeContour_Conv2D_block_v2_1.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_v2_1_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_v2_1_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_v2_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_v2_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_v2_1_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_v2_1_Conv2D_1_bias'])

                self.eyeContour_Prelu_6.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_6_weight']))
                self.eyeContour_Conv2D_block_4.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_4_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_4_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_4_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_4_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_4_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_4_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_4_Conv2D_1_bias'])

                self.eyeContour_Prelu_7.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_7_weight']))
                self.eyeContour_Conv2D_block_5.load(Conv2D_0_weight=d['eyeContour_Conv2D_block_5_Conv2D_0_weight'],Conv2D_0_bias=d['eyeContour_Conv2D_block_5_Conv2D_0_bias'], Prelu_weight=d['eyeContour_Conv2D_block_5_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["eyeContour_Conv2D_block_5_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["eyeContour_Conv2D_block_5_DepthwiseConv2d_bias"], Conv2D_1_weight=d['eyeContour_Conv2D_block_5_Conv2D_1_weight'],Conv2D_1_bias=d['eyeContour_Conv2D_block_5_Conv2D_1_bias'])

                self.eyeContour_Prelu_8.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Prelu_8_weight']))
                self.eyeContour_Conv2D_out.weight = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Conv2D_out_weight']))
                self.eyeContour_Conv2D_out.bias = torch.nn.Parameter(torch.from_numpy(d['eyeContour_Conv2D_out_bias']))


                self.iris_Conv2D_block_0.load(Conv2D_0_weight=d['iris_Conv2D_block_0_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_0_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_0_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_0_Conv2D_1_bias'])

                self.iris_Prelu_1.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_1_weight']))
                self.iris_Conv2D_block_1.load(Conv2D_0_weight=d['iris_Conv2D_block_1_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_1_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_1_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_1_Conv2D_1_bias'])

                self.iris_Prelu_2.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_2_weight']))
                self.iris_Conv2D_block_v2_0.load(Conv2D_0_weight=d['iris_Conv2D_block_v2_0_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_v2_0_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_v2_0_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_v2_0_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_v2_0_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_v2_0_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_v2_0_Conv2D_1_bias'])

                self.iris_Prelu_3.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_3_weight']))
                self.iris_Conv2D_block_2.load(Conv2D_0_weight=d['iris_Conv2D_block_2_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_2_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_2_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_2_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_2_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_2_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_2_Conv2D_1_bias'])

                self.iris_Prelu_4.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_4_weight']))
                self.iris_Conv2D_block_3.load(Conv2D_0_weight=d['iris_Conv2D_block_3_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_3_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_3_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_3_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_3_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_3_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_3_Conv2D_1_bias'])

                self.iris_Prelu_5.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_5_weight']))
                self.iris_Conv2D_block_v2_1.load(Conv2D_0_weight=d['iris_Conv2D_block_v2_1_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_v2_1_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_v2_1_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_v2_1_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_v2_1_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_v2_1_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_v2_1_Conv2D_1_bias'])

                self.iris_Prelu_6.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_6_weight']))
                self.iris_Conv2D_block_4.load(Conv2D_0_weight=d['iris_Conv2D_block_4_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_4_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_4_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_4_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_4_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_4_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_4_Conv2D_1_bias'])

                self.iris_Prelu_7.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_7_weight']))
                self.iris_Conv2D_block_5 = Conv2d_block(in_channels=128,out_channels=128, hidden_channels=64)
                self.iris_Conv2D_block_5.load(Conv2D_0_weight=d['iris_Conv2D_block_5_Conv2D_0_weight'],Conv2D_0_bias=d['iris_Conv2D_block_5_Conv2D_0_bias'], Prelu_weight=d['iris_Conv2D_block_5_Prelu_0_weight'], 
                    DepthwiseConv2d_weight=d["iris_Conv2D_block_5_DepthwiseConv2d_weight"], DepthwiseConv2d_bias=d["iris_Conv2D_block_5_DepthwiseConv2d_bias"], Conv2D_1_weight=d['iris_Conv2D_block_5_Conv2D_1_weight'],Conv2D_1_bias=d['iris_Conv2D_block_5_Conv2D_1_bias'])

                self.iris_Prelu_8.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Prelu_8_weight']))
                self.iris_Conv2D_out.weight = torch.nn.Parameter(torch.from_numpy(d['iris_Conv2D_out_weight']))
                self.iris_Conv2D_out.bias = torch.nn.Parameter(torch.from_numpy(d['iris_Conv2D_out_bias']))

                if not os.path.exists(ckpt_path):
                    torch.save({
                        "model_state_dict": self.state_dict()
                    }, ckpt_path)
                    print("Torch model saved ~")
            
            else:
                self.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
                print("Loaded from checkpoint file ~")
            
    def forward(self, x):
        x = self.padding(x)
        x = self.Conv2D_0(x)

        x = self.Prelu_0(x)
        x = x + self.Conv2D_block_0(x)
        
        x = self.Prelu_1(x)
        x = x + self.Conv2D_block_1(x)
        
        x = self.Prelu_2(x)
        x = x + self.Conv2D_block_2(x)
        
        x = self.Prelu_3(x)
        x = x + self.Conv2D_block_3(x)
        
        x = x1 = self.Prelu_4(x)
        x = self.Conv2D_block_v2_0(x, x1) 

        x = self.Prelu_5(x)
        x = x + self.Conv2D_block_4(x)

        x = self.Prelu_6(x)
        x = x + self.Conv2D_block_5(x)

        x = self.Prelu_7(x)
        x = x + self.Conv2D_block_6(x)

        x = self.Prelu_8(x)
        x = x + self.Conv2D_block_7(x)

        x = x1 = self.Prelu_9(x)
        x = self.Conv2D_block_v2_1(x, x1) 

        x_eyeContour = x_iris = self.Prelu_10(x)

        # Eye Contour
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_0(x_eyeContour)
        x_eyeContour = self.eyeContour_Prelu_1(x_eyeContour)
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_1(x_eyeContour)
        x_eyeContour = x_eyeContour_1 = self.eyeContour_Prelu_2(x_eyeContour)
        x_eyeContour = self.eyeContour_Conv2D_block_v2_0(x_eyeContour, x_eyeContour_1)
        x_eyeContour = self.eyeContour_Prelu_3(x_eyeContour)
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_2(x_eyeContour)
        x_eyeContour = self.eyeContour_Prelu_4(x_eyeContour)
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_3(x_eyeContour)
        x_eyeContour = x_eyeContour_1 = self.eyeContour_Prelu_5(x_eyeContour)
        x_eyeContour = self.eyeContour_Conv2D_block_v2_1(x_eyeContour, x_eyeContour_1)
        x_eyeContour = self.eyeContour_Prelu_6(x_eyeContour)
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_4(x_eyeContour)
        x_eyeContour = self.eyeContour_Prelu_7(x_eyeContour)
        x_eyeContour = x_eyeContour + self.eyeContour_Conv2D_block_5(x_eyeContour)
        x_eyeContour = self.eyeContour_Prelu_8(x_eyeContour)
        x_eyeContour = self.eyeContour_Conv2D_out(x_eyeContour)
        x_eyeContour = x_eyeContour.reshape((x_eyeContour.shape[0], -1))

        # Iris
        x_iris = x_iris + self.iris_Conv2D_block_0(x_iris)
        x_iris = self.iris_Prelu_1(x_iris)
        x_iris = x_iris + self.iris_Conv2D_block_1(x_iris)
        x_iris = x_iris_1 = self.iris_Prelu_2(x_iris)
        x_iris = self.iris_Conv2D_block_v2_0(x_iris, x_iris_1)
        x_iris = self.iris_Prelu_3(x_iris)
        x_iris = x_iris + self.iris_Conv2D_block_2(x_iris)
        x_iris = self.iris_Prelu_4(x_iris)
        x_iris = x_iris + self.iris_Conv2D_block_3(x_iris)
        x_iris = x_iris_1 = self.iris_Prelu_5(x_iris)
        x_iris = self.iris_Conv2D_block_v2_1(x_iris, x_iris_1)
        x_iris = self.iris_Prelu_6(x_iris)
        x_iris = x_iris + self.iris_Conv2D_block_4(x_iris)
        x_iris = self.iris_Prelu_7(x_iris)
        x_iris = x_iris + self.iris_Conv2D_block_5(x_iris)
        x_iris = self.iris_Prelu_8(x_iris)
        x_iris = self.iris_Conv2D_out(x_iris)
        x_iris = x_iris.reshape((x_iris.shape[0], -1))
        return x_eyeContour, x_iris

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == "__main__":

    with open("./data/weights.pkl", 'rb') as picklefile:
        d = pickle.load(picklefile)

    model = MediaPipeIris(pretrained=True, weights_path="./data/weights.pkl")

    eyeContour, iris = model(torch.from_numpy(d["input"]))

    err = rel_error(eyeContour.detach().numpy().flatten(), d['eyeContour_output'])
    print("Eyes Contours error: {}".format(err))

    err = rel_error(iris.detach().numpy().flatten(), d['iris_output'])
    print("Iris error: {}".format(err))

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('mediapipe model size: {:.3f}MB'.format(size_all_mb))

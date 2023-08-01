import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='softmax',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        
        self.model = smp.Unet(
                encoder_name="tu-efficientnet_b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
                aux_params=aux_params
        )
        
    def forward(self, x):
        mask, label = self.model(x)
        return mask
    
class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()
        aux_params=dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.3,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        
        self.model = smp.DeepLabV3Plus(
                encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_depth=5,
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                encoder_output_stride=16,
            
                decoder_channels=512, 
                decoder_atrous_rates=(12, 24, 36),
                in_channels=3, 
                classes=1, 
                activation="sigmoid", 
                upsampling=4,
                aux_params=aux_params
        )

    def forward(self, x):
        mask, label = self.model(x)
        return mask
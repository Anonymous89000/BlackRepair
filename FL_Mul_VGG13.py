import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback 
import random 
import time 

# ===================== VGG13 Configuration and Definition =====================
cfg = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGG13(nn.Module):
    def __init__(self, vgg_name='VGG13', nc=10):
        super(VGG13, self).__init__()
        self.in_channels = 3
        
        current_in_channels_for_block_construction = self.in_channels

        self.features1 = self._make_layers(cfg[vgg_name][0:3], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels

        self.features2 = self._make_layers(cfg[vgg_name][3:6], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels

        self.features3 = self._make_layers(cfg[vgg_name][6:9], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels

        self.features4 = self._make_layers(cfg[vgg_name][9:12], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels

        self.features5 = self._make_layers(cfg[vgg_name][12:], current_in_channels_for_block_construction)
        
        self.dense1 = nn.Linear(512, 1024) 
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, nc)

    def forward(self, x): 
        # Block 1
        conv1_1_out = self.features1[0](x)
        bn1_1_out = self.features1[1](conv1_1_out)
        f1_ = bn1_1_out # Output of features1[0:2] if [0:2] is Conv,BN

        relu1_1_out = self.features1[2](f1_)
        conv1_2_out = self.features1[3](relu1_1_out)
        bn1_2_out = self.features1[4](conv1_2_out)
        relu1_2_out = self.features1[5](bn1_2_out)
        pool1_out = self.features1[6](relu1_2_out)
        f1 = pool1_out # Output of features1[2:] applied to f1_

        # Block 2
        conv2_1_out = self.features2[0](f1)
        bn2_1_out = self.features2[1](conv2_1_out)
        f2_ = bn2_1_out

        relu2_1_out = self.features2[2](f2_)
        conv2_2_out = self.features2[3](relu2_1_out)
        bn2_2_out = self.features2[4](conv2_2_out)
        relu2_2_out = self.features2[5](bn2_2_out)
        pool2_out = self.features2[6](relu2_2_out)
        f2 = pool2_out

        # Block 3
        conv3_1_out = self.features3[0](f2)
        bn3_1_out = self.features3[1](conv3_1_out)
        f3_ = bn3_1_out

        relu3_1_out = self.features3[2](f3_)
        conv3_2_out = self.features3[3](relu3_1_out)
        bn3_2_out = self.features3[4](conv3_2_out)
        relu3_2_out = self.features3[5](bn3_2_out)
        pool3_out = self.features3[6](relu3_2_out)
        f3 = pool3_out

        # Block 4
        conv4_1_out = self.features4[0](f3)
        bn4_1_out = self.features4[1](conv4_1_out)
        f4_ = bn4_1_out
        
        relu4_1_out = self.features4[2](f4_)
        conv4_2_out = self.features4[3](relu4_1_out)
        bn4_2_out = self.features4[4](conv4_2_out)
        relu4_2_out = self.features4[5](bn4_2_out)
        pool4_out = self.features4[6](relu4_2_out)
        f4 = pool4_out

        # Block 5
        conv5_1_out = self.features5[0](f4)
        bn5_1_out = self.features5[1](conv5_1_out)
        f5_ = bn5_1_out

        relu5_1_out = self.features5[2](f5_)
        conv5_2_out = self.features5[3](relu5_1_out)
        bn5_2_out = self.features5[4](conv5_2_out)
        relu5_2_out = self.features5[5](bn5_2_out)
        pool5_out = self.features5[6](relu5_2_out)
        f5 = pool5_out
        
        f5_flat = f5.view(f5.size(0), -1) 
        
        d1_lin = self.dense1(f5_flat)
        d1 = F.relu(d1_lin) 
        
        d2_lin = self.dense2(d1)
        d2 = F.relu(d2_lin) 
        
        out = self.classifier(d2)
        return out

    def _make_layers(self, block_cfg, current_in_channels):
        layers = []
        temp_in_channels = current_in_channels
        for x_layer in block_cfg:
            if x_layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(temp_in_channels, x_layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x_layer),
                           nn.ReLU(inplace=True)] 
                temp_in_channels = x_layer
        self.in_channels = temp_in_channels 
        return nn.Sequential(*layers)

# ===================== Configuration Parameters =====================
SEED = 42
DATA_ROOT = './data' 
MODEL_DIR = 'D:/sxy/Temp/net/' 
VGG13_MODEL_FILENAMES = { 
    'raw': 'CIFAR10_innervgg13_BadNets_raw.pth',
    'bd': 'CIFAR10_innervgg13_BadNets_bd.pth' 
}

# --- Target Individual Convolutional Layers for VGG13 ---
TARGET_LAYERS_VGG13_CONV_INDIVIDUAL = [
    # Block 1
    {'id': 'f1_conv1', 'name': 'VGG13: Conv1_1 (F1[0])',  'is_conv_layer': True},
    {'id': 'f1_conv2', 'name': 'VGG13: Conv1_2 (F1[3])',  'is_conv_layer': True},
    # Block 2
    {'id': 'f2_conv1', 'name': 'VGG13: Conv2_1 (F2[0])',  'is_conv_layer': True},
    {'id': 'f2_conv2', 'name': 'VGG13: Conv2_2 (F2[3])',  'is_conv_layer': True},
    # Block 3
    {'id': 'f3_conv1', 'name': 'VGG13: Conv3_1 (F3[0])',  'is_conv_layer': True},
    {'id': 'f3_conv2', 'name': 'VGG13: Conv3_2 (F3[3])',  'is_conv_layer': True},
    # Block 4
    {'id': 'f4_conv1', 'name': 'VGG13: Conv4_1 (F4[0])',  'is_conv_layer': True},
    {'id': 'f4_conv2', 'name': 'VGG13: Conv4_2 (F4[3])',  'is_conv_layer': True},
    # Block 5
    {'id': 'f5_conv1', 'name': 'VGG13: Conv5_1 (F5[0])',  'is_conv_layer': True},
    {'id': 'f5_conv2', 'name': 'VGG13: Conv5_2 (F5[3])',  'is_conv_layer': True},
]

# --- Target Individual Fully Connected Layers for VGG13 ---
TARGET_LAYERS_VGG13_FC_INDIVIDUAL = [
    {'id': 'fc_dense1',   'name': 'VGG13: dense1 (Linear)', 'is_conv_layer': False},
    {'id': 'fc_dense2',   'name': 'VGG13: dense2 (Linear)', 'is_conv_layer': False},
    {'id': 'fc_classifier','name': 'VGG13: classifier (Linear)', 'is_conv_layer': False},
]

# --- !! User Area: Select which set of layers to analyze !! ---
# TARGET_LAYERS = TARGET_LAYERS_VGG13_CONV_INDIVIDUAL
TARGET_LAYERS = TARGET_LAYERS_VGG13_FC_INDIVIDUAL
# You can also combine them:
# TARGET_LAYERS = TARGET_LAYERS_VGG13_CONV_INDIVIDUAL + TARGET_LAYERS_VGG13_FC_INDIVIDUAL
# --- !! End User Area !! ---


BACKGROUND_SAMPLES = 50 
EXPORT_DIR_BASE = 'D:/sxy/Temp/result/vgg13_cifar10_badnets_specific_layers' 
TOP_N_LAYERS_FOR_SHAP_DISTRIBUTION = 3
TEST_CLASS_NAME = 'airplane' 
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ===================== Utility Functions =====================
def set_global_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to: {seed}")

def set_relu_inplace_false(model):
    for module_name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    return model

# ===================== Model Loader =====================
class ModelLoader:
    @staticmethod
    def load(model_type): 
        model_filename_str = VGG13_MODEL_FILENAMES.get(model_type)
        if model_filename_str is None:
            raise ValueError(f"Unknown model_type: {model_type}. Expected 'raw' or 'bd'.")
        
        model_path = os.path.join(MODEL_DIR, model_filename_str)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        
        model = VGG13(nc=10) 
        print(f"Loading VGG13 model weights from {model_path} ({model_filename_str})...")
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading state_dict file {model_path}: {e}")
            raise

        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict weight loading failed for VGG13: {e}. Attempting non-strict loading (strict=False)...")
            model.load_state_dict(new_state_dict, strict=False)
            
        print(f"Custom VGG13 Model '{model_filename_str}' (type: {model_type}) loaded successfully.")
        return model.eval()

# ===================== Backdoor Analyzer Helper Modules =====================
class VGG13ModelPart1(nn.Module):
    def __init__(self, main_model, target_id):
        super().__init__()
        self.main_model = main_model
        self.target_id = target_id
        
    def forward(self, x):
        # Block 1
        conv1_1_out = self.main_model.features1[0](x)
        if self.target_id == 'f1_conv1': return conv1_1_out
        bn1_1_out = self.main_model.features1[1](conv1_1_out)
        f1_ = bn1_1_out

        relu1_1_out = self.main_model.features1[2](f1_)
        conv1_2_out = self.main_model.features1[3](relu1_1_out)
        if self.target_id == 'f1_conv2': return conv1_2_out
        bn1_2_out = self.main_model.features1[4](conv1_2_out)
        relu1_2_out = self.main_model.features1[5](bn1_2_out)
        pool1_out = self.main_model.features1[6](relu1_2_out)
        f1 = pool1_out

        # Block 2
        conv2_1_out = self.main_model.features2[0](f1)
        if self.target_id == 'f2_conv1': return conv2_1_out
        bn2_1_out = self.main_model.features2[1](conv2_1_out)
        f2_ = bn2_1_out

        relu2_1_out = self.main_model.features2[2](f2_)
        conv2_2_out = self.main_model.features2[3](relu2_1_out)
        if self.target_id == 'f2_conv2': return conv2_2_out
        bn2_2_out = self.main_model.features2[4](conv2_2_out)
        relu2_2_out = self.main_model.features2[5](bn2_2_out)
        pool2_out = self.main_model.features2[6](relu2_2_out)
        f2 = pool2_out
        
        # Block 3
        conv3_1_out = self.main_model.features3[0](f2)
        if self.target_id == 'f3_conv1': return conv3_1_out
        bn3_1_out = self.main_model.features3[1](conv3_1_out)
        f3_ = bn3_1_out

        relu3_1_out = self.main_model.features3[2](f3_)
        conv3_2_out = self.main_model.features3[3](relu3_1_out)
        if self.target_id == 'f3_conv2': return conv3_2_out
        bn3_2_out = self.main_model.features3[4](conv3_2_out)
        relu3_2_out = self.main_model.features3[5](bn3_2_out)
        pool3_out = self.main_model.features3[6](relu3_2_out)
        f3 = pool3_out

        # Block 4
        conv4_1_out = self.main_model.features4[0](f3)
        if self.target_id == 'f4_conv1': return conv4_1_out
        bn4_1_out = self.main_model.features4[1](conv4_1_out)
        f4_ = bn4_1_out
        
        relu4_1_out = self.main_model.features4[2](f4_)
        conv4_2_out = self.main_model.features4[3](relu4_1_out)
        if self.target_id == 'f4_conv2': return conv4_2_out
        bn4_2_out = self.main_model.features4[4](conv4_2_out)
        relu4_2_out = self.main_model.features4[5](bn4_2_out)
        pool4_out = self.main_model.features4[6](relu4_2_out)
        f4 = pool4_out

        # Block 5
        conv5_1_out = self.main_model.features5[0](f4)
        if self.target_id == 'f5_conv1': return conv5_1_out
        bn5_1_out = self.main_model.features5[1](conv5_1_out)
        f5_ = bn5_1_out

        relu5_1_out = self.main_model.features5[2](f5_)
        conv5_2_out = self.main_model.features5[3](relu5_1_out)
        if self.target_id == 'f5_conv2': return conv5_2_out
        bn5_2_out = self.main_model.features5[4](conv5_2_out)
        relu5_2_out = self.main_model.features5[5](bn5_2_out)
        pool5_out = self.main_model.features5[6](relu5_2_out)
        f5 = pool5_out
        
        f5_flat = f5.view(f5.size(0), -1)
        
        d1_lin = self.main_model.dense1(f5_flat)
        if self.target_id == 'fc_dense1': return d1_lin
        d1 = F.relu(d1_lin)
        
        d2_lin = self.main_model.dense2(d1)
        if self.target_id == 'fc_dense2': return d2_lin
        d2 = F.relu(d2_lin)
        
        out_logits = self.main_model.classifier(d2)
        if self.target_id == 'fc_classifier': return out_logits
        
        raise ValueError(f"Unknown target_id for VGG13ModelPart1 (specific layers): {self.target_id}")

class VGG13ModelPart2(nn.Module):
    def __init__(self, main_model, start_from_id):
        super().__init__()
        self.main_model = main_model
        self.start_from_id = start_from_id

    def forward(self, x_intermediate): 
        inp = x_intermediate
        
        # Resume from output of a Convolutional Layer
        if self.start_from_id == 'f1_conv1': # inp is conv1_1_out
            bn1_1_out = self.main_model.features1[1](inp); f1_ = bn1_1_out
            relu1_1_out = self.main_model.features1[2](f1_)
            conv1_2_out = self.main_model.features1[3](relu1_1_out) 
            bn1_2_out = self.main_model.features1[4](conv1_2_out)
            relu1_2_out = self.main_model.features1[5](bn1_2_out)
            pool1_out = self.main_model.features1[6](relu1_2_out); inp = pool1_out # Now inp is f1
        elif self.start_from_id == 'f1_conv2': # inp is conv1_2_out
            bn1_2_out = self.main_model.features1[4](inp)
            relu1_2_out = self.main_model.features1[5](bn1_2_out)
            pool1_out = self.main_model.features1[6](relu1_2_out); inp = pool1_out # Now inp is f1
        # Link to next block if previous block finished
        if self.start_from_id in ['f1_conv1', 'f1_conv2']: # inp is f1 (output of pool1_out)
            conv2_1_out = self.main_model.features2[0](inp)
            # ... continue from conv2_1_out as if start_from_id was 'f2_conv1_input_processed'
            # This is becoming a full forward pass again, simpler to cascade
            bn2_1_out = self.main_model.features2[1](conv2_1_out); f2_ = bn2_1_out
            relu2_1_out = self.main_model.features2[2](f2_)
            conv2_2_out = self.main_model.features2[3](relu2_1_out)
            bn2_2_out = self.main_model.features2[4](conv2_2_out)
            relu2_2_out = self.main_model.features2[5](bn2_2_out)
            pool2_out = self.main_model.features2[6](relu2_2_out); inp = pool2_out # Now inp is f2
        
        if self.start_from_id == 'f2_conv1': # inp is conv2_1_out
            bn2_1_out = self.main_model.features2[1](inp); f2_ = bn2_1_out
            relu2_1_out = self.main_model.features2[2](f2_)
            conv2_2_out = self.main_model.features2[3](relu2_1_out)
            bn2_2_out = self.main_model.features2[4](conv2_2_out)
            relu2_2_out = self.main_model.features2[5](bn2_2_out)
            pool2_out = self.main_model.features2[6](relu2_2_out); inp = pool2_out
        elif self.start_from_id == 'f2_conv2': # inp is conv2_2_out
            bn2_2_out = self.main_model.features2[4](inp)
            relu2_2_out = self.main_model.features2[5](bn2_2_out)
            pool2_out = self.main_model.features2[6](relu2_2_out); inp = pool2_out
        if self.start_from_id in ['f1_conv1', 'f1_conv2', 'f2_conv1', 'f2_conv2']: # inp is f2 (output of pool2_out)
            conv3_1_out = self.main_model.features3[0](inp)
            bn3_1_out = self.main_model.features3[1](conv3_1_out); f3_ = bn3_1_out
            relu3_1_out = self.main_model.features3[2](f3_)
            conv3_2_out = self.main_model.features3[3](relu3_1_out)
            bn3_2_out = self.main_model.features3[4](conv3_2_out)
            relu3_2_out = self.main_model.features3[5](bn3_2_out)
            pool3_out = self.main_model.features3[6](relu3_2_out); inp = pool3_out # Now inp is f3

        if self.start_from_id == 'f3_conv1':
            bn3_1_out = self.main_model.features3[1](inp); f3_ = bn3_1_out
            relu3_1_out = self.main_model.features3[2](f3_)
            conv3_2_out = self.main_model.features3[3](relu3_1_out)
            bn3_2_out = self.main_model.features3[4](conv3_2_out)
            relu3_2_out = self.main_model.features3[5](bn3_2_out)
            pool3_out = self.main_model.features3[6](relu3_2_out); inp = pool3_out
        elif self.start_from_id == 'f3_conv2':
            bn3_2_out = self.main_model.features3[4](inp)
            relu3_2_out = self.main_model.features3[5](bn3_2_out)
            pool3_out = self.main_model.features3[6](relu3_2_out); inp = pool3_out
        if self.start_from_id in ['f1_conv1', 'f1_conv2', 'f2_conv1', 'f2_conv2', 'f3_conv1', 'f3_conv2']: # inp is f3
            conv4_1_out = self.main_model.features4[0](inp)
            bn4_1_out = self.main_model.features4[1](conv4_1_out); f4_ = bn4_1_out
            relu4_1_out = self.main_model.features4[2](f4_)
            conv4_2_out = self.main_model.features4[3](relu4_1_out)
            bn4_2_out = self.main_model.features4[4](conv4_2_out)
            relu4_2_out = self.main_model.features4[5](bn4_2_out)
            pool4_out = self.main_model.features4[6](relu4_2_out); inp = pool4_out # Now inp is f4

        if self.start_from_id == 'f4_conv1':
            bn4_1_out = self.main_model.features4[1](inp); f4_ = bn4_1_out
            relu4_1_out = self.main_model.features4[2](f4_)
            conv4_2_out = self.main_model.features4[3](relu4_1_out)
            bn4_2_out = self.main_model.features4[4](conv4_2_out)
            relu4_2_out = self.main_model.features4[5](bn4_2_out)
            pool4_out = self.main_model.features4[6](relu4_2_out); inp = pool4_out
        elif self.start_from_id == 'f4_conv2':
            bn4_2_out = self.main_model.features4[4](inp)
            relu4_2_out = self.main_model.features4[5](bn4_2_out)
            pool4_out = self.main_model.features4[6](relu4_2_out); inp = pool4_out
        if self.start_from_id in ['f1_conv1', 'f1_conv2', 'f2_conv1', 'f2_conv2', 'f3_conv1', 'f3_conv2', 'f4_conv1', 'f4_conv2']: # inp is f4
            conv5_1_out = self.main_model.features5[0](inp)
            bn5_1_out = self.main_model.features5[1](conv5_1_out); f5_ = bn5_1_out
            relu5_1_out = self.main_model.features5[2](f5_)
            conv5_2_out = self.main_model.features5[3](relu5_1_out)
            bn5_2_out = self.main_model.features5[4](conv5_2_out)
            relu5_2_out = self.main_model.features5[5](bn5_2_out)
            pool5_out = self.main_model.features5[6](relu5_2_out); inp = pool5_out # Now inp is f5

        if self.start_from_id == 'f5_conv1':
            bn5_1_out = self.main_model.features5[1](inp); f5_ = bn5_1_out
            relu5_1_out = self.main_model.features5[2](f5_)
            conv5_2_out = self.main_model.features5[3](relu5_1_out)
            bn5_2_out = self.main_model.features5[4](conv5_2_out)
            relu5_2_out = self.main_model.features5[5](bn5_2_out)
            pool5_out = self.main_model.features5[6](relu5_2_out); inp = pool5_out
        elif self.start_from_id == 'f5_conv2':
            bn5_2_out = self.main_model.features5[4](inp)
            relu5_2_out = self.main_model.features5[5](bn5_2_out)
            pool5_out = self.main_model.features5[6](relu5_2_out); inp = pool5_out
        # After all feature blocks, inp is f5 (output of pool5_out)

        f5_flat = inp.view(inp.size(0), -1)
        inp = f5_flat # Now inp is f5_flat
        
        # Resume from output of an FC Layer (these are simpler)
        if self.start_from_id == 'fc_dense1': # inp is d1_lin (output of dense1)
            d1 = F.relu(inp); inp = d1
        elif self.start_from_id == 'fc_dense2': # inp is d2_lin (output of dense2)
            d2 = F.relu(inp); inp = d2
        elif self.start_from_id == 'fc_classifier': # inp is logits (output of classifier)
            return inp # Nothing more to do
        else: # Came from conv layers, inp is f5_flat
            d1_lin = self.main_model.dense1(inp)
            d1 = F.relu(d1_lin); inp = d1

        # If execution reached here, inp is d1
        if self.start_from_id != 'fc_dense2' and self.start_from_id != 'fc_classifier': # Don't re-evaluate if starting from dense2 or classifier
            d2_lin = self.main_model.dense2(inp)
            d2 = F.relu(d2_lin); inp = d2
        
        # If execution reached here, inp is d2
        if self.start_from_id != 'fc_classifier': # Don't re-evaluate if starting from classifier
            out_logits = self.main_model.classifier(inp)
            return out_logits
        
        # Should have returned by now if start_from_id was valid and not 'fc_classifier'
        # This path is for when start_from_id itself was 'fc_classifier', handled above.
        # Or if an invalid ID somehow reached here.
        raise ValueError(f"Unhandled continuation logic in VGG13ModelPart2 for start_from_id: {self.start_from_id}")


# ===================== Backdoor Analyzer =====================
class BackdoorAnalyzer:
    def _parse_attack_type_from_filename(self, filename_str):
        try:
            name_part = os.path.splitext(os.path.basename(filename_str))[0] 
            parts = name_part.split('_') 
            for i in range(len(parts) -1, 0, -1):
                if parts[i].lower() == "bd" or parts[i].lower() == "raw":
                    if i > 0 and parts[i-1].lower() not in ["cifar10", "innervgg13", "vgg13", "class10"]:
                        return parts[i-1] 
            if "bd" in name_part.lower(): return "backdoor" 
            relevant_parts = [p for p in parts if p.lower() not in ["cifar10", "class10"]]
            if relevant_parts:
                return "_".join(relevant_parts)[:20]
            return name_part[:20] 
        except Exception:
            return "unknown_attack"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        print("Loading Raw VGG13 Model...")
        self.base_raw_model = ModelLoader.load('raw').to(self.device)
        self.base_raw_model = set_relu_inplace_false(self.base_raw_model)
        print("Raw VGG13 Model ReLU inplace operations disabled.")
        
        self.attacked_model_filename = VGG13_MODEL_FILENAMES['bd']
        self.parsed_attack_type = self._parse_attack_type_from_filename(self.attacked_model_filename)
        print(f"Parsed attack type from '{self.attacked_model_filename}': {self.parsed_attack_type}")
        
        print(f"Loading Attacked VGG13 Model ({self.parsed_attack_type})...")
        self.base_bd_model = ModelLoader.load('bd').to(self.device)
        self.base_bd_model = set_relu_inplace_false(self.base_bd_model)
        print("Attacked VGG13 Model ReLU inplace operations disabled.")
        
        print("Preparing CIFAR10 data...")
        self.dataset, self.test_img, self.test_class_idx = self._load_data()
        print(f"Using BACKGROUND_SAMPLES: {BACKGROUND_SAMPLES}")
        self.background = self._prepare_background()
        
        print(f"Test image shape: {self.test_img.shape}, Target Class Index: {self.test_class_idx} ({CIFAR10_CLASSES[self.test_class_idx]})")
        print(f"Background data shape: {self.background.shape}")
        print(f"Dataset classes: {CIFAR10_CLASSES}")

    def _load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        try:
            val_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
        except Exception as e:
            print(f"Error downloading/loading CIFAR10 dataset from {DATA_ROOT}: {e}")
            raise
            
        if not val_dataset.classes or len(val_dataset.classes) != 10:
            raise ValueError(f"CIFAR10 did not load correctly.")

        try:
            target_idx = CIFAR10_CLASSES.index(TEST_CLASS_NAME)
        except ValueError:
            raise ValueError(f"Test class '{TEST_CLASS_NAME}' not found in CIFAR10 classes. Available: {CIFAR10_CLASSES}")

        class_indices_in_dataset = [i for i, label in enumerate(val_dataset.targets) if label == target_idx]
        if not class_indices_in_dataset:
            raise ValueError(f"No samples found for test class '{TEST_CLASS_NAME}' (index {target_idx}) in CIFAR10 test set.")
        
        test_img_tensor, _ = val_dataset[class_indices_in_dataset[0]]
        return val_dataset, test_img_tensor.unsqueeze(0).to(self.device), target_idx

    def _prepare_background(self):
        if not self.dataset: raise ValueError("Dataset not loaded, cannot prepare background.")
        num_to_take = min(BACKGROUND_SAMPLES, len(self.dataset))
        if len(self.dataset) < BACKGROUND_SAMPLES and BACKGROUND_SAMPLES > 0:
            print(f"Warning: Dataset size ({len(self.dataset)}) is smaller than BACKGROUND_SAMPLES ({BACKGROUND_SAMPLES}). Using {num_to_take} samples.")
        elif BACKGROUND_SAMPLES == 0:
            if len(self.dataset) > 0:
                print("Warning: BACKGROUND_SAMPLES is 0. Using the first dataset sample as background.")
                return self.dataset[0][0].unsqueeze(0).to(self.device)
            else: raise ValueError("Dataset is empty and BACKGROUND_SAMPLES is 0. Cannot create background data.")
        
        allow_replace = num_to_take > len(self.dataset)
        indices = np.random.choice(len(self.dataset), num_to_take, replace=allow_replace)
        return torch.stack([self.dataset[i][0] for i in indices]).to(self.device)

    def _get_model_parts(self, base_model, target_layer_info):
        base_model.eval() 
        target_id = target_layer_info['id']

        model_part1_module = VGG13ModelPart1(base_model, target_id).to(self.device).eval()
        
        # If targeting the final classifier output, model_part2 is Identity
        if target_id == 'fc_classifier': 
            model_part2_module = nn.Identity().to(self.device).eval()
        else:
            model_part2_module = VGG13ModelPart2(base_model, target_id).to(self.device).eval()
            
        return model_part1_module, model_part2_module

    def analyze(self):
        analysis_start_time_sec = time.time()
        analysis_start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(analysis_start_time_sec))
        print(f"\n--- Overall Layer Contribution Analysis (VGG13 on CIFAR10 - Specific Layers) started at: {analysis_start_time_str} ---")

        if not TARGET_LAYERS:
            print("Warning: TARGET_LAYERS is empty. No analysis points.")
            return None
        
        raw_shap_values_for_viz = []
        bd_shap_values_for_viz = []
        layer_contributions_list = []
        num_classes = len(CIFAR10_CLASSES)

        for target_layer_info in TARGET_LAYERS:
            layer_name_for_print = target_layer_info['name']
            target_id = target_layer_info['id']
            is_conv_target_layer = target_layer_info.get('is_conv_layer', False) # Get flag from new TARGET_LAYERS
            print(f"\nAnalyzing Target: '{layer_name_for_print}' (ID: {target_id})...")
            
            model_part1_raw, model_part2_raw = self._get_model_parts(self.base_raw_model, target_layer_info)
            model_part1_bd, model_part2_bd = self._get_model_parts(self.base_bd_model, target_layer_info)

            with torch.no_grad():
                activations_test_raw = model_part1_raw(self.test_img)
                original_bg_activations_raw = model_part1_raw(self.background)
                activations_test_bd = model_part1_bd(self.test_img)
                original_bg_activations_bd = model_part1_bd(self.background)

            _shap_input_activations_test_raw = activations_test_raw
            _shap_input_activations_background_raw = original_bg_activations_raw
            _shap_input_activations_test_bd = activations_test_bd
            _shap_input_activations_background_bd = original_bg_activations_bd
            
            # FC-like layers (is_conv_target_layer is False) may benefit from background averaging for DeepExplainer
            if not is_conv_target_layer: 
                if _shap_input_activations_background_raw.shape[0] > 1:
                    _shap_input_activations_background_raw = _shap_input_activations_background_raw.mean(dim=0, keepdim=True)
                if _shap_input_activations_background_bd.shape[0] > 1:
                    _shap_input_activations_background_bd = _shap_input_activations_background_bd.mean(dim=0, keepdim=True)

            if _shap_input_activations_background_raw.numel() == 0:
                 raise ValueError(f"Background data for SHAP (Raw) resulted in an empty tensor (Layer: {layer_name_for_print}).")
            if _shap_input_activations_background_bd.numel() == 0:
                 raise ValueError(f"Background data for SHAP (Attacked) resulted in an empty tensor (Layer: {layer_name_for_print}).")

            try:
                model_part2_raw.eval(); model_part2_bd.eval()
                # Use GradientExplainer for Convolutional layer outputs, DeepExplainer for FC layer outputs
                explainer_type = shap.GradientExplainer if is_conv_target_layer else shap.DeepExplainer
                explainer_args = {'local_smoothing': 0.0} if explainer_type == shap.GradientExplainer else {}
                
                bg_data_raw_device = _shap_input_activations_background_raw.to(self.device)
                explainer_raw = explainer_type(model_part2_raw, bg_data_raw_device, **explainer_args)
                shap_raw_list_or_arr = explainer_raw.shap_values(_shap_input_activations_test_raw.to(self.device))

                bg_data_bd_device = _shap_input_activations_background_bd.to(self.device)
                explainer_bd = explainer_type(model_part2_bd, bg_data_bd_device, **explainer_args)
                shap_bd_list_or_arr = explainer_bd.shap_values(_shap_input_activations_test_bd.to(self.device))

                def process_shap_output(shap_output, num_cls, layer_name_for_err): # Same as before
                    if isinstance(shap_output, list):
                        if len(shap_output) == num_cls: return shap_output
                        else:
                            processed_list = []
                            for item in shap_output:
                                if isinstance(item, np.ndarray) and item.ndim > 0 and item.shape[-1] == num_cls:
                                    for c_idx in range(num_cls): processed_list.append(item[...,c_idx])
                                else: processed_list.append(item)
                            if len(processed_list) == num_cls: return processed_list
                            raise ValueError(f"SHAP format error (list) for {layer_name_for_err}. Len mismatch. Got {len(shap_output)}")
                    elif isinstance(shap_output, np.ndarray):
                        if shap_output.ndim >= 1 and shap_output.shape[-1] == num_cls:
                            squeezed_shap = shap_output.squeeze(0) if shap_output.shape[0] == 1 and shap_output.ndim > 1 else shap_output
                            return [squeezed_shap[..., i] for i in range(num_cls)]
                        else: raise ValueError(f"SHAP format error (array) for {layer_name_for_err}. Unexpected shape: {shap_output.shape}")
                    else: raise ValueError(f"Unexpected SHAP output type for {layer_name_for_err}: {type(shap_output)}")

                shap_raw_list = process_shap_output(shap_raw_list_or_arr, num_classes, layer_name_for_print + " (Raw)")
                shap_bd_list = process_shap_output(shap_bd_list_or_arr, num_classes, layer_name_for_print + " (Attacked)")
                
                num_activations = np.prod(activations_test_raw.shape[1:]) 
                proc_shap_raw = np.zeros((int(num_activations), num_classes)); proc_shap_bd = np.zeros((int(num_activations), num_classes)) # Ensure int
                
                for c in range(num_classes):
                    cur_raw = shap_raw_list[c]; cur_bd = shap_bd_list[c]
                    if isinstance(cur_raw, torch.Tensor): cur_raw = cur_raw.cpu().numpy()
                    if isinstance(cur_bd, torch.Tensor): cur_bd = cur_bd.cpu().numpy()
                    
                    if cur_raw.size != num_activations:
                        raise ValueError(f"SHAP value size mismatch for Raw, Class {c}, Layer {layer_name_for_print}. Expected {num_activations}, got {cur_raw.size}. SHAP shape: {cur_raw.shape}, Activ shape: {activations_test_raw.shape[1:]}")
                    if cur_bd.size != num_activations:
                         raise ValueError(f"SHAP value size mismatch for Attacked, Class {c}, Layer {layer_name_for_print}. Expected {num_activations}, got {cur_bd.size}. SHAP shape: {cur_bd.shape}, Activ shape: {activations_test_bd.shape[1:]}")

                    proc_shap_raw[:,c] = cur_raw.reshape(-1)
                    proc_shap_bd[:,c] = cur_bd.reshape(-1)

                raw_shap_values_for_viz.append(proc_shap_raw); bd_shap_values_for_viz.append(proc_shap_bd)
                diff_abs_shap = np.abs(np.abs(proc_shap_bd) - np.abs(proc_shap_raw))
                layer_contributions_list.append(np.mean(diff_abs_shap)) 
                print(f"  Contribution for '{layer_name_for_print}': {layer_contributions_list[-1]:.6f}")

            except Exception as e_shap:
                print(f"  Error during SHAP for layer '{layer_name_for_print}': {e_shap}"); traceback.print_exc()
                layer_contributions_list.append(np.nan)
                num_act_fallback = np.prod(activations_test_raw.shape[1:]) if activations_test_raw is not None and activations_test_raw.nelement() > 0 else 1
                dummy_shap = np.full((int(num_act_fallback), num_classes), np.nan)
                raw_shap_values_for_viz.append(dummy_shap); bd_shap_values_for_viz.append(dummy_shap)
            
            if 'explainer_raw' in locals(): del explainer_raw
            if 'explainer_bd' in locals(): del explainer_bd
            del model_part1_raw, model_part2_raw, model_part1_bd, model_part2_bd
            del activations_test_raw, activations_test_bd, original_bg_activations_raw, original_bg_activations_bd
            if self.device.type == 'cuda': torch.cuda.empty_cache()

        analysis_end_time_sec = time.time()
        # ... rest of the analyze method is the same, returning results ...
        analysis_end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(analysis_end_time_sec))
        analysis_duration_seconds = analysis_end_time_sec - analysis_start_time_sec
        minutes = int(analysis_duration_seconds // 60); seconds = analysis_duration_seconds % 60
        analysis_duration_str = f"{minutes} min {seconds:.2f} sec" if minutes > 0 else f"{seconds:.2f} sec"
        print(f"--- Overall Layer Contribution Analysis (VGG13 on CIFAR10 - Specific Layers) finished at: {analysis_end_time_str}, Duration: {analysis_duration_str} ---")
        
        return {
            'layer_contributions': layer_contributions_list, 
            'class_names': CIFAR10_CLASSES, 
            'raw_shap_values': raw_shap_values_for_viz, 
            'bd_shap_values': bd_shap_values_for_viz,
            'target_layers_info': TARGET_LAYERS, 
            'parsed_attack_type': self.parsed_attack_type,
            'analysis_start_time_str': analysis_start_time_str, 
            'analysis_end_time_str': analysis_end_time_str,
            'analysis_duration_str': analysis_duration_str,
            'test_class_name': TEST_CLASS_NAME, 
            'test_class_idx': self.test_class_idx
        }


# ===================== Visualization and Export System =====================
# ResultVisualizer class remains the same as in the previous version,
# as its plotting logic is general enough. Ensure plot titles and labels remain in English.
class ResultVisualizer:
    @staticmethod
    def _get_base_filename_parts(results): 
        attack_type = results.get('parsed_attack_type', 'unknown').replace(" ", "_").replace(":", "_")
        # Modify tag slightly if analyzing specific conv/fc layers
        analysis_scope_tag = "specific_layers_vgg13" 
        current_target_layers = results.get('target_layers_info', [])
        if current_target_layers:
            if all(tl.get('is_conv_layer', False) for tl in current_target_layers):
                analysis_scope_tag = "CONV_layers_vgg13"
            elif all(not tl.get('is_conv_layer', True) for tl in current_target_layers): # if all are FC
                analysis_scope_tag = "FC_layers_vgg13"
        
        if len(current_target_layers) == 1:
            name_to_sanitize = current_target_layers[0]['name']
            analysis_scope_tag = name_to_sanitize.replace(" ","_").replace("(","").replace(")","").replace(":","").replace("/","-").replace("[","").replace("]","")


        class_name_simple = results.get('test_class_name', 'unknown_class').replace("/", "-").replace("\\", "-")
        return attack_type, analysis_scope_tag, class_name_simple

    @staticmethod
    def plot(results, save_dir): 
        if not results or 'layer_contributions' not in results: print("Warning: No layer contribution data to plot."); return
        attack_type, analysis_scope_fn_tag, cls_name_fn = ResultVisualizer._get_base_filename_parts(results)
        
        lc = results['layer_contributions']; tl_info = results.get('target_layers_info', [])
        valid_idx = [i for i, x in enumerate(lc) if not (isinstance(x,float) and np.isnan(x))]
        if not valid_idx: print("Warning: All contributions NaN. Cannot plot contributions bar chart."); return
        
        valid_contrib = [lc[i] for i in valid_idx]
        y_labels = [tl_info[i].get('name', f'Layer {tl_info[i].get("id", i)}') for i in valid_idx]
        if not y_labels: print("Warning: No valid labels for plotting contributions."); return
        
        fig, ax = plt.subplots(figsize=(14, max(6, 2 + 0.6 * len(valid_contrib)))) 
        y_pos = np.arange(len(valid_contrib))
        bars = ax.barh(y_pos, valid_contrib, height=0.5, color='#2E86C1', edgecolor='black', alpha=0.8)
        
        max_v_val = 0
        if valid_contrib:
            numeric_contrib = [c for c in valid_contrib if isinstance(c, (int, float)) and not np.isnan(c)]
            if numeric_contrib:
                max_v_val = np.max(numeric_contrib) if np.max(numeric_contrib) > 0 else 0

        offset = (max_v_val * 0.02) if max_v_val > 0 else 0.00002 
        
        for bar_patch in bars:
            val = bar_patch.get_width()
            ax.text(val + offset, bar_patch.get_y() + bar_patch.get_height()/2, f"{val:.4f}", va='center', ha='left', fontsize=8)
        
        ax.set_yticks(y_pos); ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel(f'Layer Contribution (Mean Abs. SHAP Diff. Raw vs. {attack_type})', fontsize=11) # English
        
        # Determine scope title based on actual layers analyzed
        scope_title = "Contribution of Analyzed Layers in VGG13" # Default English
        current_target_layers = results.get('target_layers_info', [])
        if current_target_layers:
            if all(tl.get('is_conv_layer', False) for tl in current_target_layers):
                scope_title = "Contribution of Convolutional Layers in VGG13"
            elif all(not tl.get('is_conv_layer', True) for tl in current_target_layers):
                 scope_title = "Contribution of Fully-Connected Layers in VGG13"

        ax.set_title(f'{scope_title}\n(Test Class: {results.get("test_class_name", "N/A")})', fontsize=13, pad=10) # English
        ax.invert_yaxis(); ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        fig.tight_layout(rect=[0,0,1,0.96])
        # Adjust left margin based on label length
        max_label_len = max(len(L) for L in y_labels) if y_labels else 0
        adj_left = 0.45 if max_label_len > 40 else (0.38 if max_label_len > 30 else 0.30)
        plt.subplots_adjust(left=adj_left)
        
        filepath = os.path.join(save_dir, f"layer_contributions_{attack_type}_{analysis_scope_fn_tag}_cls{cls_name_fn}.pdf")
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=300); print(f"Plot saved: {filepath}")
        except Exception as e: print(f"Error saving plot {filepath}: {e}")
        plt.show(); plt.close(fig)

    @staticmethod
    def visualize_shap_distributions(results, save_dir): 
        if not results or 'raw_shap_values' not in results : print("Warning: SHAP values missing for distribution plots."); return
        attack_type, analysis_scope_fn_tag, cls_name_fn = ResultVisualizer._get_base_filename_parts(results)
        
        raw_list = results['raw_shap_values']; bd_list = results['bd_shap_values']
        class_names = results['class_names'] 
        tl_info = results['target_layers_info']
        if not raw_list or not bd_list or not tl_info or not class_names:
            print("Warning: SHAP data, target layer info, or class names empty for distributions."); return
        
        num_cls = len(class_names)
        if num_cls == 0: print("Warning: Class names empty for distributions."); return

        contributions = results['layer_contributions']
        layers_with_data_and_scores = []
        for i in range(len(tl_info)):
            if i < len(raw_list) and i < len(bd_list) and \
               isinstance(raw_list[i], np.ndarray) and isinstance(bd_list[i], np.ndarray) and \
               not np.all(np.isnan(raw_list[i])) and not np.all(np.isnan(bd_list[i])) and \
               i < len(contributions) and not (isinstance(contributions[i], float) and np.isnan(contributions[i])):
                layers_with_data_and_scores.append({
                    'original_index': i,
                    'contribution': contributions[i],
                    'name': tl_info[i].get('name', f'Point {tl_info[i].get("id", i)}'),
                    'raw_shap': raw_list[i],
                    'bd_shap': bd_list[i]
                })
        
        sorted_layers = sorted(layers_with_data_and_scores, key=lambda x: x['contribution'], reverse=True)
        top_n_layers_to_plot_info = sorted_layers[:TOP_N_LAYERS_FOR_SHAP_DISTRIBUTION]
        
        if not top_n_layers_to_plot_info:
            print("Warning: No valid layers with contributions to plot for SHAP distributions after filtering."); return
        
        num_layers_to_plot = len(top_n_layers_to_plot_info)
        fig, axes = plt.subplots(num_layers_to_plot, 2, figsize=(15, max(6, 5.5 * num_layers_to_plot)), squeeze=False)
        
        suptitle_y = 0.98 if num_layers_to_plot <= 1 else 0.97 
        if num_layers_to_plot > 2 : suptitle_y = 0.985 
        tight_layout_top = suptitle_y - 0.04

        fig.suptitle(f'Top {num_layers_to_plot} Layers: SHAP Value Distributions (Raw vs. {attack_type} Attack)\nTest Class: {results.get("test_class_name", "N/A")}', 
                       fontsize=14, y=suptitle_y) # English

        for plot_r_idx, layer_data in enumerate(top_n_layers_to_plot_info):
            l_name = layer_data['name']
            raw_d = layer_data['raw_shap']; bd_d = layer_data['bd_shap']
            
            ax_r = axes[plot_r_idx,0]; ax_b = axes[plot_r_idx,1]
            
            data_r_plot = [raw_d[:,c] for c in range(num_cls) if c < raw_d.shape[1] and not np.all(np.isnan(raw_d[:,c]))]
            pos_r = [c for c in range(num_cls) if c < raw_d.shape[1] and not np.all(np.isnan(raw_d[:,c]))]
            valid_class_names_r = [class_names[p] for p in pos_r]

            if data_r_plot: 
                ax_r.boxplot(data_r_plot, positions=pos_r, showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='#ADD8E6',alpha=0.7), medianprops=dict(color='red'))
            ax_r.set_title(f"{l_name}\nRaw Model SHAP Dist.", fontsize=10, pad=10) # English
            ax_r.set_xticks(pos_r); ax_r.set_xticklabels(valid_class_names_r, rotation=45, ha="right", fontsize=8)
            ax_r.set_ylabel('SHAP Value', fontsize=9); ax_r.grid(axis='y', linestyle='--', alpha=0.7) # English

            data_b_plot = [bd_d[:,c] for c in range(num_cls) if c < bd_d.shape[1] and not np.all(np.isnan(bd_d[:,c]))]
            pos_b = [c for c in range(num_cls) if c < bd_d.shape[1] and not np.all(np.isnan(bd_d[:,c]))]
            valid_class_names_b = [class_names[p] for p in pos_b]

            if data_b_plot: 
                ax_b.boxplot(data_b_plot, positions=pos_b, showfliers=False, widths=0.6, patch_artist=True, boxprops=dict(facecolor='#FFB6C1',alpha=0.7), medianprops=dict(color='blue'))
            ax_b.set_title(f"{l_name}\n{attack_type} Model SHAP Dist.", fontsize=10, pad=10) # English
            ax_b.set_xticks(pos_b); ax_b.set_xticklabels(valid_class_names_b, rotation=45, ha="right", fontsize=8)
            ax_b.set_ylabel('SHAP Value', fontsize=9); ax_b.grid(axis='y', linestyle='--', alpha=0.7) # English
        
        plt.tight_layout(rect=[0,0.03,1, tight_layout_top])
        filepath = os.path.join(save_dir, f"shap_distributions_TOP{num_layers_to_plot}_{attack_type}_{analysis_scope_fn_tag}_cls{cls_name_fn}.pdf")
        try:
            fig.savefig(filepath, bbox_inches='tight', dpi=300); print(f"Plot saved: {filepath}")
        except Exception as e: print(f"Error saving plot {filepath}: {e}")
        plt.show(); plt.close(fig)


# ===================== Main Execution Block =====================
if __name__ == "__main__":
    set_global_seeds(SEED)
    os.makedirs(EXPORT_DIR_BASE, exist_ok=True)
    
    analysis_scope_desc_en = "VGG13 Specific Layers (CIFAR10)" 
    if TARGET_LAYERS == TARGET_LAYERS_VGG13_CONV_INDIVIDUAL:
        analysis_scope_desc_en = "VGG13 Individual Convolutional Layers (CIFAR10)"
    elif TARGET_LAYERS == TARGET_LAYERS_VGG13_FC_INDIVIDUAL:
        analysis_scope_desc_en = "VGG13 Individual Fully-Connected Layers (CIFAR10)"


    print(f"""
    =============================================================
    Custom VGG13 (CIFAR10) Specific Layer Contribution Analysis
    (Code comments and plot texts are in English)
    =============================================================
    """)
    overall_start_time = time.time()
    try:
        analyzer = BackdoorAnalyzer()
        results = analyzer.analyze()
        if results:
            print(f"\n--- Analysis Summary: {analysis_scope_desc_en} ---")
            test_class_display = results.get('test_class_name', 'N/A')
            print(f"Primary Test Class: {test_class_display} (Index: {results.get('test_class_idx', 'N/A')})")
            print(f"Parsed Attack Type for BD Model: {results.get('parsed_attack_type', 'N/A')}")
            print(f"Analysis Phase Duration: {results.get('analysis_duration_str', 'N/A')}")
            
            if 'target_layers_info' in results and 'layer_contributions' in results:
                print(f"Contribution of each analyzed layer:")
                for info, contrib in zip(results['target_layers_info'], results['layer_contributions']):
                    name = info.get('name', f"ID: {info.get('id', 'Unknown')}") 
                    if isinstance(contrib, float) and np.isnan(contrib): print(f"  - {name}: Calculation Failed (NaN)")
                    else:
                        try: print(f"  - {name}: {contrib:.6f}")
                        except TypeError: print(f"  - {name}: {contrib} (Cannot format)")
            else: print("Could not retrieve detailed layer contribution data.")
            
            print("\nGenerating and saving visualizations...")
            ResultVisualizer.plot(results, EXPORT_DIR_BASE)
            ResultVisualizer.visualize_shap_distributions(results, EXPORT_DIR_BASE)
            
        else: print("Analysis did not return results. Cannot visualize.")
        
    except FileNotFoundError as e: print(f"\nERROR - File Not Found: {e}")
    except ValueError as e: print(f"\nERROR - Value/Configuration Issue: {e}\nTraceback:"); traceback.print_exc()
    except IndexError as e: print(f"\nERROR - Index Issue: {e}\nTraceback:"); traceback.print_exc()
    except RuntimeError as e:
        print(f"\nERROR - Runtime Issue: {e}")
        if "CUDA out of memory" in str(e): print("CUDA out of memory. Try reducing BACKGROUND_SAMPLES or check GPU.")
        traceback.print_exc()
    except Exception as e: print(f"\nAn unexpected error occurred: {e}"); traceback.print_exc()
    finally:
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        minutes = int(overall_duration // 60); seconds = overall_duration % 60
        overall_duration_str = f"{minutes} min {seconds:.2f} sec" if minutes > 0 else f"{seconds:.2f} sec"
        print(f"\nTotal script execution time: {overall_duration_str}")
        print("=============================================================")
        print("VGG13 (CIFAR10) Specific Layer Contribution Analysis System finished.")
        print(f"Results (plots as PDF) should be saved in: {EXPORT_DIR_BASE}")
        print("=============================================================")
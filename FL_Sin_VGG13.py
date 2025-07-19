import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback # For more detailed error printing
import random # For setting Python's built-in random seed
import time # For recording time

# ===================== VGG13 Model Definition =====================
cfg = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

class VGG13(nn.Module):
    def __init__(self, vgg_name='VGG13', nc=10):
        super(VGG13, self).__init__()
        self.in_channels = 3
        
        current_in_channels_for_block_construction = 3
        self.features1 = self._make_layers(cfg[vgg_name][0:3], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels
        self.features2 = self._make_layers(cfg[vgg_name][3:6], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels
        self.features3 = self._make_layers(cfg[vgg_name][6:9], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels
        self.features4 = self._make_layers(cfg[vgg_name][9:12], current_in_channels_for_block_construction)
        current_in_channels_for_block_construction = self.in_channels
        self.features5 = self._make_layers(cfg[vgg_name][12:], current_in_channels_for_block_construction)
        
        self.dense1 = nn.Linear(512 * 1 * 1, 1024) # For CIFAR-10 32x32 input, after 5 MaxPools, size is 1x1
        self.dense2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, nc)

    def forward(self, x): 
        f1 = self.features1(x)
        f2 = self.features2(f1)
        f3 = self.features3(f2)
        f4 = self.features4(f3)
        f5 = self.features5(f4)
        f5_flat = f5.view(f5.size(0), -1) 
        d1_lin = self.dense1(f5_flat)
        d1 = F.relu(d1_lin) 
        d2_lin = self.dense2(d1)
        d2 = F.relu(d2_lin) 
        out = self.classifier(d2)
        return out

    def _make_layers(self, block_cfg, current_in_channels_for_this_block):
        layers = []
        temp_in_channels = current_in_channels_for_this_block
        for x_layer in block_cfg:
            if x_layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(temp_in_channels, x_layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x_layer), # Added BatchNorm
                           nn.ReLU(inplace=True)] 
                temp_in_channels = x_layer
        self.in_channels = temp_in_channels # Update self.in_channels for the next block
        return nn.Sequential(*layers)

_captured_layer_input_global = None

def _forward_hook_capture_input_fn(module, input_tensor_tuple, output_tensor):
    global _captured_layer_input_global
    _captured_layer_input_global = input_tensor_tuple[0].clone().detach()

# ===================== Configuration Parameters =====================
SEED = 42
DATA_ROOT = './data/cifar10' 
MODEL_DIR = 'D:/sxy/Temp/net/' # 请确保路径正确
RAW_MODEL_FILENAME = 'CIFAR10_innervgg13_WaNet_raw.pth'#Blended WaNet BadNets
ATTACKED_MODEL_FILENAME = 'CIFAR10_innervgg13_WaNet_bd.pth'

ANALYSIS_MODE = 'fc'
# CONV_TARGET_SPEC = {'block_attr': 'features4', 'layer_idx': 0}#conv4_1
# CONV_TARGET_SPEC = {'block_attr': 'features5', 'layer_idx': 0}#conv5_1
CONV_TARGET_SPEC = {'block_attr': 'features5', 'layer_idx': 3}# conv5_2
FC_TARGET_ATTR_NAME = 'dense2' 

BACKGROUND_SAMPLES = 50 
TEST_CLASS_NAME = 'airplane' # CIFAR-10 class name
TOP_NEURONS_SUMMARY = 10     
TOP_CHANNELS_SUMMARY = 10   
TOP_NEURONS_DETAILED_PLOT = 5 
TOP_CHANNELS_DETAILED_PLOT = 3 
TOP_NEURONS_EXPORT = 20 # Max items prepared by analyze() for export
TOP_K_PARAM_CONTRIBS_PER_ACTIVATION = 5

EXPORT_DIR_BASE = 'D:/sxy/Temp/result/vgg13_cifar10_single_714-1' # 请确保路径正确
CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
EXPORT_PARAMETERS = True # Set to True to export layer parameters
PARAMS_EXPORT_SUBDIR = "exported_parameters" 

# ===================== Utility Functions =====================
def set_global_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to: {seed}")

def set_relu_inplace_false(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.ReLU): module.inplace = False
    return model

# ===================== Model Loader =====================
class ModelLoader:
    @staticmethod
    def load(model_filename_str, model_dir):
        model_path = os.path.join(model_dir, model_filename_str)
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file {model_path} not found")
        model = VGG13(nc=len(CIFAR10_CLASSES)) # Use VGG13 for CIFAR-10
        print(f"Loading VGG13 model weights from {model_path} ({model_filename_str})...")
        state_dict = torch.load(model_path, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
        
        # Adjust keys if necessary (e.g. remove 'module.' prefix)
        new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
        
        try: model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict weight loading failed: {e}. Attempting non-strict loading for VGG13...")
            model.load_state_dict(new_state_dict, strict=False) # Allow non-strict for flexibility
        model = set_relu_inplace_false(model)
        print(f"Model {model_filename_str} (VGG13) loaded successfully, and ReLU.inplace set to False.")
        return model.eval()

# ===================== VGG13 Specific Model Parts for SHAPWrapper =====================
class _VGG13ProcessingPart1(nn.Module): # Computes up to the target layer's output
    def __init__(self, main_model, target_id): # target_id like 'fX_convY' or 'fc_denseZ'
        super().__init__()
        self.main_model = main_model
        self.target_id = target_id
        
    def forward(self, x):
        # Iteratively process features blocks
        f_out = x
        for i in range(1, 6): # features1 to features5
            block_attr_name = f"features{i}"
            current_block = getattr(self.main_model, block_attr_name)
            # Process layers within the block (Conv, BN, ReLU, Conv, BN, ReLU, Pool)
            # Target IDs: f{i}_conv1 (after current_block[0]), f{i}_conv2 (after current_block[3])
            
            # Conv1 in block i
            if hasattr(current_block, '0') and isinstance(current_block[0], nn.Conv2d):
                conv1_out = current_block[0](f_out)
                if self.target_id == f"f{i}_conv1": return conv1_out
                if hasattr(current_block, '1') and hasattr(current_block, '2'):
                     bn1_out = current_block[1](conv1_out)
                     relu1_out = current_block[2](bn1_out)
                else: raise AttributeError(f"Missing BN/ReLU after conv1 in {block_attr_name}")
            else: raise AttributeError(f"Missing conv1 in {block_attr_name}")

            # Conv2 in block i
            if hasattr(current_block, '3') and isinstance(current_block[3], nn.Conv2d):
                conv2_out = current_block[3](relu1_out)
                if self.target_id == f"f{i}_conv2": return conv2_out
                if hasattr(current_block, '4') and hasattr(current_block, '5'):
                    bn2_out = current_block[4](conv2_out)
                    relu2_out = current_block[5](bn2_out)
                else: raise AttributeError(f"Missing BN/ReLU after conv2 in {block_attr_name}")
            else: raise AttributeError(f"Missing conv2 in {block_attr_name}")
            
            # MaxPool at end of block i
            if hasattr(current_block, '6') and isinstance(current_block[6], nn.MaxPool2d):
                f_out = current_block[6](relu2_out) # Output of current block is input to next
            else: raise AttributeError(f"Missing MaxPool in {block_attr_name}")

            if i == 5: # After features5 block, before flattening
                break # f_out is now output of features5's MaxPool

        # FC layers
        f_flat = f_out.view(f_out.size(0), -1)
        
        d1_lin = self.main_model.dense1(f_flat)
        if self.target_id == 'fc_dense1': return d1_lin
        d1_act = F.relu(d1_lin)
        
        d2_lin = self.main_model.dense2(d1_act)
        if self.target_id == 'fc_dense2': return d2_lin
        d2_act = F.relu(d2_lin)
        
        out_logits = self.main_model.classifier(d2_act)
        if self.target_id == 'fc_classifier': return out_logits
        
        raise ValueError(f"_VGG13ProcessingPart1: Unknown or unreached target_id: {self.target_id}")


class _VGG13ProcessingPart2(nn.Module): # Computes from target layer's output to final model output
    def __init__(self, main_model, start_from_id): # start_from_id is target_id of Part1
        super().__init__()
        self.main_model = main_model
        self.start_from_id = start_from_id
    
    def forward(self, x_intermediate): # x_intermediate is the output of Part1
        current_data = x_intermediate
        
        start_block_num = 0
        is_fc_start = self.start_from_id.startswith('fc_')
        
        if not is_fc_start: # Convolutional layer start
            block_char = self.start_from_id[1] # e.g., 'f4_conv1' -> '4'
            conv_num_in_block_char = self.start_from_id[-1] # e.g., 'f4_conv1' -> '1'
            start_block_num = int(block_char) # e.g., 4
            
            # Finish current block if needed
            current_block_module = getattr(self.main_model, f"features{start_block_num}")
            if self.start_from_id == f"f{start_block_num}_conv1": # Output of Conv1, need BN1,ReLU1,Conv2,BN2,ReLU2,Pool
                bn1_out = current_block_module[1](current_data)
                relu1_out = current_block_module[2](bn1_out)
                conv2_out = current_block_module[3](relu1_out)
                bn2_out = current_block_module[4](conv2_out)
                relu2_out = current_block_module[5](bn2_out)
                current_data = current_block_module[6](relu2_out) # Pool output
            elif self.start_from_id == f"f{start_block_num}_conv2": # Output of Conv2, need BN2,ReLU2,Pool
                bn2_out = current_block_module[4](current_data)
                relu2_out = current_block_module[5](bn2_out)
                current_data = current_block_module[6](relu2_out) # Pool output
            
            # Process subsequent feature blocks
            for next_block_idx in range(start_block_num + 1, 6): # e.g., if start_block_num=4, loop for 5
                current_data = getattr(self.main_model, f"features{next_block_idx}")(current_data)
            
            # Flatten for FC layers
            current_data = current_data.view(current_data.size(0), -1)
            if current_data.shape[1] != 512 : # VGG13 specific for CIFAR-10 after features
                 raise RuntimeError(f"Shape mismatch before dense1: {current_data.shape}, expected 512 for VGG13 on CIFAR-10. Start ID: {self.start_from_id}")

            # Pass through all FC layers as we are now at the input of dense1
            d1_lin = self.main_model.dense1(current_data)
            d1_act = F.relu(d1_lin)
            d2_lin = self.main_model.dense2(d1_act)
            d2_act = F.relu(d2_lin)
            out = self.main_model.classifier(d2_act)
            return out

        else: # FC layer start
            if self.start_from_id == 'fc_dense1': # x_intermediate is d1_lin (output of dense1 before ReLU)
                d1_act = F.relu(current_data)
                d2_lin = self.main_model.dense2(d1_act)
                d2_act = F.relu(d2_lin)
                out = self.main_model.classifier(d2_act)
                return out
            elif self.start_from_id == 'fc_dense2': # x_intermediate is d2_lin
                d2_act = F.relu(current_data)
                out = self.main_model.classifier(d2_act)
                return out
            elif self.start_from_id == 'fc_classifier': # x_intermediate is output of classifier (logits)
                return current_data # This is the final output
        
        raise ValueError(f"_VGG13ProcessingPart2: Unhandled continuation for start_from_id: {self.start_from_id}")

class VGG13SHAPWrapper(nn.Module):
    def __init__(self, base_model, analysis_mode='conv',
                 conv_target_spec=None, fc_target_attr_name=None): # conv_target_spec = {'block_attr': 'featuresX', 'layer_idx': Y}
        super().__init__()
        self.base_model = base_model # Keep a reference
        self.analysis_mode = analysis_mode
        self.target_layer_output_shape = None
        self.conceptual_layer_name = "Unknown"
        self.part1_target_id = None # Will be like 'fX_convY' or 'fc_denseZ'

        model_device = next(base_model.parameters()).device

        if self.analysis_mode == 'conv':
            if conv_target_spec is None: raise ValueError("conv_target_spec (dict) required for 'conv' mode.")
            block_attr = conv_target_spec['block_attr'] # e.g. 'features4'
            layer_idx_in_block = conv_target_spec['layer_idx']   # 0 for first Conv2d, 3 for second Conv2d in VGG13 block
            
            # Determine conceptual conv number in block (1st or 2nd)
            if layer_idx_in_block == 0: conv_num_in_block = 1
            elif layer_idx_in_block == 3: conv_num_in_block = 2
            else: raise ValueError(f"For VGG13, conv_target_spec['layer_idx'] must be 0 (1st conv) or 3 (2nd conv). Got {layer_idx_in_block}.")
            
            block_num_char = block_attr[-1] # From 'featuresX'
            self.part1_target_id = f"f{block_num_char}_conv{conv_num_in_block}" # e.g. f4_conv1
            self.conceptual_layer_name = f"VGG13 {block_attr}[Conv{conv_num_in_block} at idx {layer_idx_in_block}] (target_id: {self.part1_target_id})"

        elif self.analysis_mode == 'fc':
            if fc_target_attr_name is None: raise ValueError("fc_target_attr_name (str) required for 'fc' mode.")
            if fc_target_attr_name not in ['dense1', 'dense2', 'classifier']:
                raise ValueError(f"Invalid fc_target_attr_name for VGG13: {fc_target_attr_name}. Must be 'dense1', 'dense2', or 'classifier'.")
            self.part1_target_id = f"fc_{fc_target_attr_name}" # e.g. fc_dense2
            self.conceptual_layer_name = f"VGG13 {fc_target_attr_name} (target_id: {self.part1_target_id})"
        else:
            raise ValueError(f"Unsupported analysis_mode: {self.analysis_mode}")

        self.features_part1 = _VGG13ProcessingPart1(base_model, self.part1_target_id)
        
        # Part2 starts processing AFTER the target layer's output (which is x_intermediate)
        # If target is the final classifier output, part2 is just identity.
        if self.analysis_mode == 'fc' and self.part1_target_id == 'fc_classifier':
            self.features_part2 = nn.Identity()
        else:
            self.features_part2 = _VGG13ProcessingPart2(base_model, self.part1_target_id)
            
        # Determine target_layer_output_shape using a dummy input
        with torch.no_grad():
            # CIFAR-10 images are 3x32x32
            dummy_input = torch.randn(1, 3, 32, 32, device=model_device) 
            out_features_part1 = self.features_part1(dummy_input)
            self.target_layer_output_shape = out_features_part1.shape[1:] # Exclude batch dim
        print(f"VGG13SHAPWrapper ({self.analysis_mode} Mode): Target for SHAP: '{self.conceptual_layer_name}', Output shape from part1: {self.target_layer_output_shape}")

    def forward(self, x): # x is the original image input
        x_part1_out = self.features_part1(x) # Output of the target layer
        return self.features_part2(x_part1_out) # Output of the rest of the model


# ===================== Backdoor Analyzer (VGG13 adapted) =====================
class BackdoorAnalyzer:
    def _parse_attack_type_from_filename(self, filename_str):
        try:
            name_part = os.path.splitext(os.path.basename(filename_str))[0]
            parts = name_part.split('_')
            # Look for "bd" or "raw" and take the part before it, if not common model/dataset names
            for i in range(len(parts) - 1, 0, -1): # Iterate backwards
                if parts[i].lower() == "bd" or parts[i].lower() == "raw":
                    if i > 0 and parts[i-1].lower() not in ["cifar10", "innervgg13", "vgg13", "class10"]:
                        return parts[i-1] # Attack type is likely the part before _bd or _raw
            # Fallback: if "bd" is in the name, assume a generic backdoor type if specific one not found
            if "bd" in name_part.lower(): return "backdoor" # Generic if specific not parsed
            # Further fallback: try to clean common terms and take what's left
            relevant_parts = [p for p in parts if p.lower() not in ["cifar10", "class10", "innervgg13", "vgg13"]]
            if relevant_parts: return "_".join(relevant_parts)[:20] # Take first 20 chars of remaining
            return name_part[:20] # Last resort: take first 20 chars of filename part
        except Exception:
            return "unknown_attack"


    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"\nUsing device: {self.device}")
        self.analysis_mode = ANALYSIS_MODE

        # Load original VGG13 models (not SHAP wrappers yet)
        self.base_raw_model_orig = ModelLoader.load(RAW_MODEL_FILENAME, MODEL_DIR).to(self.device)
        self.base_bd_model_orig = ModelLoader.load(ATTACKED_MODEL_FILENAME, MODEL_DIR).to(self.device)

        self.parsed_attack_type = self._parse_attack_type_from_filename(ATTACKED_MODEL_FILENAME)
        print(f"Parsed attack type from '{ATTACKED_MODEL_FILENAME}': {self.parsed_attack_type}")

        # Create SHAP wrappers
        if self.analysis_mode == 'conv':
            self.raw_model_wrapper = VGG13SHAPWrapper(self.base_raw_model_orig, 'conv', conv_target_spec=CONV_TARGET_SPEC).to(self.device).eval()
            self.bd_model_wrapper = VGG13SHAPWrapper(self.base_bd_model_orig, 'conv', conv_target_spec=CONV_TARGET_SPEC).to(self.device).eval()
        elif self.analysis_mode == 'fc':
            self.raw_model_wrapper = VGG13SHAPWrapper(self.base_raw_model_orig, 'fc', fc_target_attr_name=FC_TARGET_ATTR_NAME).to(self.device).eval()
            self.bd_model_wrapper = VGG13SHAPWrapper(self.base_bd_model_orig, 'fc', fc_target_attr_name=FC_TARGET_ATTR_NAME).to(self.device).eval()
        else:
            raise ValueError(f"Unsupported ANALYSIS_MODE: {self.analysis_mode}")
        
        self.conceptual_layer_name = self.raw_model_wrapper.conceptual_layer_name # Get from wrapper
        self.target_layer_identifier = self.conceptual_layer_name # Use the same descriptive name
        self.target_layer_actual_output_shape = self.raw_model_wrapper.target_layer_output_shape

        self.dataset, self.test_img, self.test_class_idx = self._load_data()
        self.background = self._prepare_background()
        print(f"Test img shape: {self.test_img.shape}, Bg shape: {self.background.shape}, Test class: {CIFAR10_CLASSES[self.test_class_idx]} (idx {self.test_class_idx})")


    def _load_data(self, num_test_images=1):
        # Using CIFAR-10 specific normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        try:
            # Ensure data root exists, download if necessary
            if not os.path.exists(DATA_ROOT) or not os.listdir(DATA_ROOT): # Check if dir exists and is not empty
                print(f"Data root {DATA_ROOT} not found or empty. Attempting CIFAR10 download to this location.")
                os.makedirs(DATA_ROOT, exist_ok=True) # Create dir if it doesn't exist
                # Download train and test to ensure DATA_ROOT/cifar-10-batches-py/ exists
                datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
                datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform) # Also download train
            dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform) # Load test set
        except Exception as e:
            raise FileNotFoundError(f"CIFAR10 dataset download or loading failed at {DATA_ROOT}. Error: {e}")
        
        # CIFAR10_CLASSES is defined globally
        if not dataset.classes or len(dataset.classes) != len(CIFAR10_CLASSES):
            print("Dataset classes not matching global CIFAR10_CLASSES, overriding.")
            dataset.classes = list(CIFAR10_CLASSES) # Ensure consistency
            
        try:
            target_idx = CIFAR10_CLASSES.index(TEST_CLASS_NAME)
        except ValueError:
            raise ValueError(f"Class name '{TEST_CLASS_NAME}' not found in CIFAR10_CLASSES tuple.")
        
        # Get indices of all samples belonging to the target class
        class_indices = [i for i, label in enumerate(dataset.targets) if label == target_idx]

        if not class_indices:
            raise ValueError(f"No samples found for class '{TEST_CLASS_NAME}'.")
        
        # Select the first sample of the target class for testing
        # You could add logic here to select multiple or random samples if needed
        test_img_tensor, _ = dataset[class_indices[0]] 
        return dataset, test_img_tensor.unsqueeze(0).to(self.device), target_idx

    def _prepare_background(self):
        print(f"Preparing {BACKGROUND_SAMPLES} SHAP background samples...")
        num_to_take = min(BACKGROUND_SAMPLES, len(self.dataset))
        if len(self.dataset) < BACKGROUND_SAMPLES and BACKGROUND_SAMPLES > 0:
            print(f"Warning: Dataset size ({len(self.dataset)}) is less than BACKGROUND_SAMPLES ({BACKGROUND_SAMPLES}). Using {num_to_take} samples.")
        elif BACKGROUND_SAMPLES == 0: 
            if len(self.dataset) > 0: return self.dataset[0][0].unsqueeze(0).to(self.device)
            else: raise ValueError("Dataset is empty and BACKGROUND_SAMPLES is 0. Cannot prepare background.")
        
        # Ensure enough samples for random choice without replacement if possible
        replace_flag = len(self.dataset) < num_to_take
        indices = np.random.choice(len(self.dataset), num_to_take, replace=replace_flag)
        return torch.stack([self.dataset[i][0] for i in indices]).to(self.device)

    def _analyze_conv_activation_to_weights_contribution_vgg13(
        self, base_model_obj, conv_target_spec, # conv_target_spec = {'block_attr': 'featuresX', 'layer_idx': Y}
        target_activation_coords_list, test_input_image, top_k_contribs_to_show
    ):
        global _captured_layer_input_global
        all_activations_detailed_contribs = []

        block_attr = conv_target_spec['block_attr']
        layer_idx_in_block = conv_target_spec['layer_idx'] # This is the index within the nn.Sequential of the block

        try:
            target_block_module = getattr(base_model_obj, block_attr) # e.g., model.features4
            target_conv_layer = target_block_module[layer_idx_in_block] # e.g., model.features4[0] or model.features4[3]
            if not isinstance(target_conv_layer, nn.Conv2d):
                print(f"Error: Layer {block_attr}[{layer_idx_in_block}] is not nn.Conv2d. It's {type(target_conv_layer)}."); return [[] for _ in target_activation_coords_list]
        except (AttributeError, IndexError) as e:
            print(f"Error accessing target conv layer {block_attr}[{layer_idx_in_block}]: {e}"); return [[] for _ in target_activation_coords_list]

        hook_handle = target_conv_layer.register_forward_hook(_forward_hook_capture_input_fn)
        base_model_obj.eval()
        with torch.no_grad(): _ = base_model_obj(test_input_image.to(self.device)) # Run model to trigger hook
        hook_handle.remove()

        if _captured_layer_input_global is None:
            print("Error: Failed to capture input to the target convolutional layer (VGG13)."); _captured_layer_input_global = None; return [[] for _ in target_activation_coords_list]
        
        layer_input_tensor = _captured_layer_input_global[0].cpu(); _captured_layer_input_global = None # Reset global
        C_in_actual, H_in_actual, W_in_actual = layer_input_tensor.shape
        K_h, K_w = target_conv_layer.kernel_size
        S_h, S_w = target_conv_layer.stride
        P_h, P_w = target_conv_layer.padding

        for C_out, H_out, W_out in target_activation_coords_list: # Coords of the target output activation
            current_activation_contribs_list = []
            if not (0 <= C_out < target_conv_layer.out_channels):
                print(f"Skipping contribution analysis for activation ({C_out},{H_out},{W_out}): Output channel {C_out} out of bounds for layer with {target_conv_layer.out_channels} channels.")
                all_activations_detailed_contribs.append(current_activation_contribs_list)
                continue
            
            filter_weights = target_conv_layer.weight.data[C_out, :, :, :].cpu() # Get weights for the specific output channel
            input_patch_for_filter = torch.zeros_like(filter_weights) 
            
            for c_in_idx in range(C_in_actual): 
                for k_h_idx in range(K_h):       
                    for k_w_idx in range(K_w):   
                        # Calculate corresponding input coordinates for this filter weight
                        h_on_input = H_out * S_h - P_h + k_h_idx
                        w_on_input = W_out * S_w - P_w + k_w_idx
                        if 0 <= h_on_input < H_in_actual and 0 <= w_on_input < W_in_actual:
                            input_patch_for_filter[c_in_idx, k_h_idx, k_w_idx] = layer_input_tensor[c_in_idx, h_on_input, w_on_input]
            
            element_wise_products = filter_weights * input_patch_for_filter 
            contributions_flat = element_wise_products.flatten().numpy()
            abs_contributions_flat = np.abs(contributions_flat)
            sorted_contrib_indices_flat = np.argsort(abs_contributions_flat)[::-1]

            for k_loop_idx in range(min(top_k_contribs_to_show, len(sorted_contrib_indices_flat))):
                flat_idx = sorted_contrib_indices_flat[k_loop_idx]
                # Convert flat index back to 3D index (c_in_filter, k_h_filter, k_w_filter)
                c_in_f, k_h_f, k_w_f = np.unravel_index(flat_idx, filter_weights.shape)
                weight_val = filter_weights[c_in_f, k_h_f, k_w_f].item()
                input_val = input_patch_for_filter[c_in_f, k_h_f, k_w_f].item() 
                contrib_prod_val = element_wise_products[c_in_f, k_h_f, k_w_f].item()
                current_activation_contribs_list.append({
                    'weight_coord_in_filter': (c_in_f, k_h_f, k_w_f), # (InputChannelToFilter, KernelH, KernelW)
                    'weight_val': weight_val, 'input_val': input_val, 'contrib_prod': contrib_prod_val 
                })
            all_activations_detailed_contribs.append(current_activation_contribs_list)
        return all_activations_detailed_contribs

    def analyze(self):
        id_phase_start_time_sec = time.time()
        def get_shap(shap_wrapper_part1, shap_wrapper_part2, bg_imgs, test_img_single):
            shap_wrapper_part1.eval(); shap_wrapper_part2.eval()
            with torch.no_grad():
                bg_dev, test_dev = bg_imgs.to(self.device), test_img_single.to(self.device)
                background_activations = shap_wrapper_part1(bg_dev)
                test_activations = shap_wrapper_part1(test_dev)
            
            # SHAP Explainer selection
            explainer_class = shap.DeepExplainer if self.analysis_mode == 'fc' else shap.GradientExplainer
            explainer_args = {} if self.analysis_mode == 'fc' else {'local_smoothing': 0.0}
            print(f"  Using shap.{explainer_class.__name__} for {self.analysis_mode} mode (VGG13)...")
            
            # Ensure part2 is on the same device as its input (background_activations)
            explainer = explainer_class(shap_wrapper_part2.to(self.device), background_activations.to(self.device), **explainer_args)
            
            print(f"  Calculating SHAP contributions for target layer test activations (VGG13)...")
            shap_all_classes = explainer.shap_values(test_activations.to(self.device)) # test_activations should also be on device

            # Handle SHAP output format (list for multi-output, or numpy array)
            if isinstance(shap_all_classes, list): # Typically for multi-output models
                shap_values_for_class = shap_all_classes[self.test_class_idx]
            elif isinstance(shap_all_classes, np.ndarray) and shap_all_classes.ndim > 0 :
                # Check if last dim is num_classes and select for target_class_idx
                if shap_all_classes.ndim > 1 and shap_all_classes.shape[-1] == len(CIFAR10_CLASSES): # Assuming last dim is classes
                     shap_values_for_class = shap_all_classes[..., self.test_class_idx] # Take slice for target class
                else: # Assume it's already for a single output or class-agnostic contributions to a specific output
                     shap_values_for_class = shap_all_classes
            else:
                raise TypeError(f"Unexpected SHAP values output type: {type(shap_all_classes)}")

            if isinstance(shap_values_for_class, torch.Tensor): 
                shap_values_for_class = shap_values_for_class.cpu().numpy()
            
            # SHAP output might have an extra batch dimension of 1, remove if so,
            # but only if its shape is (1, C, H, W) and target is (C,H,W) for example
            if shap_values_for_class.ndim > 0 and shap_values_for_class.shape[0] == 1 and \
               len(shap_values_for_class.shape) > len(self.target_layer_actual_output_shape):
                return shap_values_for_class[0] 
            return shap_values_for_class


        print(f"\nAnalyzing Raw Model ({RAW_MODEL_FILENAME}) (VGG13)..."); 
        raw_shap = get_shap(self.raw_model_wrapper.features_part1, self.raw_model_wrapper.features_part2, self.background, self.test_img)
        
        print(f"\nAnalyzing Attacked Model ({ATTACKED_MODEL_FILENAME}, type: {self.parsed_attack_type}) (VGG13)..."); 
        bd_shap = get_shap(self.bd_model_wrapper.features_part1, self.bd_model_wrapper.features_part2, self.background, self.test_img)
        
        print(f"VGG13 Raw SHAP output shape: {raw_shap.shape}, BD SHAP output shape: {bd_shap.shape}")
        if raw_shap.shape != self.target_layer_actual_output_shape:
            print(f"Warning: Raw SHAP shape {raw_shap.shape} does not match target layer output shape {self.target_layer_actual_output_shape}")
        if bd_shap.shape != self.target_layer_actual_output_shape:
            print(f"Warning: BD SHAP shape {bd_shap.shape} does not match target layer output shape {self.target_layer_actual_output_shape}")


        neuron_abs_shap_raw = np.abs(raw_shap); neuron_abs_shap_bd = np.abs(bd_shap)
        neuron_diff = np.abs(neuron_abs_shap_bd - neuron_abs_shap_raw) # Difference of absolute SHAP values
        neuron_diff_flattened = neuron_diff.flatten()

        # Sort all neurons by difference
        all_sorted_flat_indices = np.argsort(neuron_diff_flattened)[::-1]
        
        # Select top N for export (based on global TOP_NEURONS_EXPORT, now 20 for VGG13)
        num_items_to_prepare = min(TOP_NEURONS_EXPORT, len(all_sorted_flat_indices))
        export_top_n_flat_indices = all_sorted_flat_indices[:num_items_to_prepare]
        export_top_n_diff_values = neuron_diff_flattened[export_top_n_flat_indices]
        
        neurons_for_export = [] # This list will hold (coords, diff_val, detailed_contribs)
        
        if self.analysis_mode == 'conv':
            C, H, W = self.target_layer_actual_output_shape
            export_unflat_coords_conv = [np.unravel_index(idx, (C,H,W)) for idx in export_top_n_flat_indices]

            print(f"\nPerforming detailed weight-input contribution analysis for top {len(export_unflat_coords_conv)} CONV activations (VGG13)...")
            # Use the VGG13 specific contribution analysis
            detailed_contrib_lists_for_export = self._analyze_conv_activation_to_weights_contribution_vgg13(
                self.base_bd_model_orig, # Analyze attacked model's parameters
                CONV_TARGET_SPEC,       # VGG13 specific target spec
                export_unflat_coords_conv, 
                self.test_img, 
                TOP_K_PARAM_CONTRIBS_PER_ACTIVATION
            )
            for i in range(len(export_unflat_coords_conv)):
                unflat_idx = export_unflat_coords_conv[i]
                diff_val = export_top_n_diff_values[i]
                detailed_contribs = detailed_contrib_lists_for_export[i] if detailed_contrib_lists_for_export and i < len(detailed_contrib_lists_for_export) else []
                neurons_for_export.append(((unflat_idx), diff_val, detailed_contribs))
        
        else: # FC mode
            export_unflat_coords_fc = [(idx,) for idx in export_top_n_flat_indices] # FC indices are 1D
            for i in range(len(export_unflat_coords_fc)):
                unflat_idx = export_unflat_coords_fc[i]
                diff_val = export_top_n_diff_values[i]
                neurons_for_export.append(((unflat_idx), diff_val, [])) # No detailed_contribs for FC here
        
        # For console summary and detailed plots (uses different TOP_N settings)
        top_n_flat_summary_indices = all_sorted_flat_indices[:min(TOP_NEURONS_SUMMARY, len(all_sorted_flat_indices))]
        top_n_diff_val_summary = neuron_diff_flattened[top_n_flat_summary_indices]
        
        top_n_flat_detailed_indices = all_sorted_flat_indices[:min(TOP_NEURONS_DETAILED_PLOT, len(all_sorted_flat_indices))]
        
        if self.analysis_mode == 'conv':
            C,H,W = self.target_layer_actual_output_shape # Ensure C,H,W are defined for conv
            top_n_unflat_summary = [np.unravel_index(idx, (C,H,W)) for idx in top_n_flat_summary_indices]
            top_n_unflat_detailed = [np.unravel_index(idx, (C,H,W)) for idx in top_n_flat_detailed_indices]
        else: # FC
            top_n_unflat_summary = [(idx,) for idx in top_n_flat_summary_indices]
            top_n_unflat_detailed = [(idx,) for idx in top_n_flat_detailed_indices]

        # Get SHAP values for the detailed plot neurons from the full SHAP arrays
        # Ensure indices are valid if flat_detailed_indices can be empty
        top_n_raw_shap_detailed = raw_shap.flatten()[top_n_flat_detailed_indices] if top_n_flat_detailed_indices.size > 0 else np.array([])
        top_n_bd_shap_detailed = bd_shap.flatten()[top_n_flat_detailed_indices] if top_n_flat_detailed_indices.size > 0 else np.array([])
        
        # Channel-level analysis for CONV mode
        top_channel_results = {}
        if self.analysis_mode == 'conv':
            # Sum absolute SHAP values across spatial dimensions for each channel
            ch_total_abs_raw = np.sum(np.abs(raw_shap), axis=(1,2)) # raw_shap is (C,H,W)
            ch_total_abs_bd = np.sum(np.abs(bd_shap), axis=(1,2))
            ch_diff = np.abs(ch_total_abs_bd - ch_total_abs_raw) # Difference of these channel sums
            
            sorted_ch_indices = np.argsort(ch_diff)[::-1]
            top_ch_indices_summary = sorted_ch_indices[:min(TOP_CHANNELS_SUMMARY, len(sorted_ch_indices))]
            top_ch_diff_values_summary = ch_diff[top_ch_indices_summary]
            top_ch_indices_detailed_plot = sorted_ch_indices[:min(TOP_CHANNELS_DETAILED_PLOT, len(sorted_ch_indices))]
            top_channel_results = {
                'top_channel_diff_values_summary': top_ch_diff_values_summary,
                'top_channels_indices_summary': top_ch_indices_summary,
                'top_channels_indices_detailed': top_ch_indices_detailed_plot,
            }
        
        id_phase_duration_str = f"{(time.time() - id_phase_start_time_sec):.2f} sec"
        print(f"--- Neuron/Channel Identification and Parameter Contribution Analysis Phase Duration (VGG13): {id_phase_duration_str} ---")

        results_dict = {
            'analysis_mode': self.analysis_mode,
            'test_class_name': CIFAR10_CLASSES[self.test_class_idx], # Use CIFAR10 class name
            'target_layer_actual_output_shape': self.target_layer_actual_output_shape,
            'conceptual_layer_name': self.conceptual_layer_name, # From SHAP Wrapper
            'target_layer_identifier': self.target_layer_identifier, # From SHAP Wrapper
            'parsed_attack_type': self.parsed_attack_type,
            'identification_start_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(id_phase_start_time_sec)), # Added
            'identification_end_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), # Added
            'identification_duration_str': id_phase_duration_str,
            'top_neuron_diff_values_summary': top_n_diff_val_summary,
            'top_neurons_unflat_indices_summary': top_n_unflat_summary,
            'top_neurons_unflat_indices_detailed': top_n_unflat_detailed,
            'top_neurons_raw_shap_values': top_n_raw_shap_detailed,
            'top_neurons_bd_shap_values': top_n_bd_shap_detailed,
            'neurons_for_export': neurons_for_export, # This is the primary list for TXT export
            'raw_shap_at_target_layer': raw_shap,
            'bd_shap_at_target_layer': bd_shap,
            **top_channel_results # Add channel results if conv mode
        }
        return results_dict

    def access_fc_neuron_parameters_vgg13(self, model_name, model_obj, fc_attr_name, neuron_idx_tuples):
        if self.analysis_mode != 'fc': return
        if not hasattr(model_obj, fc_attr_name): 
            print(f"Model '{model_name}' does not have attribute '{fc_attr_name}'."); return
        
        fc_layer = getattr(model_obj, fc_attr_name)
        if not isinstance(fc_layer, nn.Linear): 
            print(f"Attribute '{fc_attr_name}' in '{model_name}' is not an nn.Linear layer. Type: {type(fc_layer)}"); return
        
        print(f"\n--- Accessing Parameters for Neurons in {fc_attr_name.upper()} of model '{model_name}' ---")
        print(f"Target FC Layer Module Details: {fc_layer}")
        print(f"Layer weights shape (out_features, in_features): {fc_layer.weight.shape}")
        if fc_layer.bias is not None:
            print(f"Layer bias shape (out_features): {fc_layer.bias.shape}")
        else:
            print("Layer has no bias.")

        if not neuron_idx_tuples: print("No neuron indices provided for inspection."); return
        
        print(f"Note: The following neuron indices are 0-based direct indices of the *output neurons* of the {fc_attr_name.upper()} layer.")
        for neuron_idx_tuple in neuron_idx_tuples:
            if not neuron_idx_tuple: continue 
            neuron_idx = neuron_idx_tuple[0] 
            if not (0 <= neuron_idx < fc_layer.out_features):
                print(f"  Neuron index {neuron_idx} is out of bounds for layer with {fc_layer.out_features} output neurons.")
                continue
            with torch.no_grad(): 
                neuron_weights = fc_layer.weight.data[neuron_idx, :] 
                bias_value_str = "N/A (no bias)"
                if fc_layer.bias is not None:
                    bias_value_str = f"{fc_layer.bias.data[neuron_idx].item():.6f}"
            print(f"  {fc_attr_name.upper()} Layer - Neuron Index: {neuron_idx}")
            print(f"    Bias Value: {bias_value_str}")
            print(f"    Weights Vector Shape (connecting to all inputs of this neuron): {neuron_weights.shape}")
            print(f"    First 5 values of Weights Vector: {neuron_weights[:5].cpu().numpy()}")
            print("-" * 20)


# ===================== Parameter Export Functions (VGG13 adapted) =====================
def get_module_from_path_vgg13(model, path_spec): # path_spec can be CONV_TARGET_SPEC or FC_TARGET_ATTR_NAME
    module = model
    if isinstance(path_spec, dict): # Conv mode: {'block_attr': 'featuresX', 'layer_idx': Y}
        block_module = getattr(model, path_spec['block_attr'])
        module = block_module[path_spec['layer_idx']]
    elif isinstance(path_spec, str): # FC mode: 'denseX' or 'classifier'
        module = getattr(model, path_spec)
    else:
        raise TypeError(f"Unsupported path_spec type for VGG13: {type(path_spec)}")
    return module

def export_suspicious_layer_parameters(analysis_results, raw_model, bd_model, base_export_dir):
    conceptual_name = analysis_results.get('conceptual_layer_name', 'unknown_layer_conceptual')
    
    target_spec_for_path = None
    file_id_prefix = "target_layer"

    if ANALYSIS_MODE == 'conv':
        target_spec_for_path = CONV_TARGET_SPEC # e.g. {'block_attr': 'features5', 'layer_idx': 3}
        file_id_prefix = f"{CONV_TARGET_SPEC['block_attr']}_conv_idx{CONV_TARGET_SPEC['layer_idx']}"
    elif ANALYSIS_MODE == 'fc':
        target_spec_for_path = FC_TARGET_ATTR_NAME # e.g. 'dense2'
        file_id_prefix = FC_TARGET_ATTR_NAME
    
    if not target_spec_for_path:
        print("Warning: No target specification found for parameter export."); return

    export_path_full = os.path.join(base_export_dir, PARAMS_EXPORT_SUBDIR)
    os.makedirs(export_path_full, exist_ok=True)
    
    print(f"\n--- Exporting Parameters for Target Layer: {conceptual_name} ---")
    print(f"    Parameters will be saved in: {export_path_full}")
    
    try:
        raw_target_module = get_module_from_path_vgg13(raw_model, target_spec_for_path)
        bd_target_module = get_module_from_path_vgg13(bd_model, target_spec_for_path)
        
        attack_type_str = analysis_results.get('parsed_attack_type', 'unknown_attack_type')

        # Export raw model parameters
        if hasattr(raw_target_module, 'weight') and raw_target_module.weight is not None:
            torch.save(raw_target_module.weight.data.cpu(), os.path.join(export_path_full, f"{file_id_prefix}_{attack_type_str}_raw_weights.pt"))
        if hasattr(raw_target_module, 'bias') and raw_target_module.bias is not None:
            torch.save(raw_target_module.bias.data.cpu(), os.path.join(export_path_full, f"{file_id_prefix}_{attack_type_str}_raw_bias.pt"))

        # Export attacked (bd) model parameters
        if hasattr(bd_target_module, 'weight') and bd_target_module.weight is not None:
            torch.save(bd_target_module.weight.data.cpu(), os.path.join(export_path_full, f"{file_id_prefix}_{attack_type_str}_bd_weights.pt"))
        if hasattr(bd_target_module, 'bias') and bd_target_module.bias is not None:
            torch.save(bd_target_module.bias.data.cpu(), os.path.join(export_path_full, f"{file_id_prefix}_{attack_type_str}_bd_bias.pt"))
            
        print(f"    Successfully exported parameters for '{conceptual_name}'.")

    except Exception as e:
        print(f"ERROR during parameter export for '{conceptual_name}': {e}")
        traceback.print_exc()
    print(f"--- Parameter Export Finished ---")


# ===================== Visualization and Export System (VGG13 adapted) =====================
class ResultVisualizer:
    @staticmethod
    def _get_base_filename_parts(results): # Used for plots and potentially TXT file name components
        attack_type = results.get('parsed_attack_type','unknown_attack').replace(" ","_").replace(":","_")
        # Clean up conceptual_layer_name for use in filenames
        layer_name_cleaned = results.get('conceptual_layer_name','layer_unknown').replace(" ","_").replace("[","").replace("]","").replace("(","").replace(")","").replace("_output","").replace(":","").replace(".","_").replace("/","_")
        class_name_cleaned = TEST_CLASS_NAME.replace("/","-").replace("\\","-") # Use TEST_CLASS_NAME for CIFAR10
        return attack_type, layer_name_cleaned, class_name_cleaned

    @staticmethod
    def export_neuron_analysis_to_file(results, base_dir):
        # === MODIFIED SECTION START ===
        all_items_to_consider = results.get('neurons_for_export', []) 
        num_available_items_from_analysis = len(all_items_to_consider)

        filtered_items_to_export = []
        analysis_mode = results['analysis_mode']
        
        export_condition_str_fn = "" 
        header_export_condition_line = "" 

        if analysis_mode == 'conv':
            CONV_EXPORT_TARGET_COUNT = 20 # Directly take top 20 for CONV mode (no threshold)

            num_conv_to_actually_export = min(CONV_EXPORT_TARGET_COUNT, num_available_items_from_analysis)
            filtered_items_to_export = all_items_to_consider[:num_conv_to_actually_export]
            
            if not filtered_items_to_export:
                print(f"No CONV items available for export (source list was empty or target count was 0).")
            else:
                print(f"CONV mode: Exporting top {num_conv_to_actually_export} items (targeted up to {CONV_EXPORT_TARGET_COUNT}, {num_available_items_from_analysis} available from analysis step).")

            export_condition_str_fn = f"_top{num_conv_to_actually_export}conv" 
            header_export_condition_line = f"Export Condition (Conv): Top {num_conv_to_actually_export} items (targeted up to {CONV_EXPORT_TARGET_COUNT} from {num_available_items_from_analysis} available)\n"
        
        elif analysis_mode == 'fc':
            FC_EXPORT_TARGET_COUNT = 30 
            num_fc_to_actually_export = min(FC_EXPORT_TARGET_COUNT, num_available_items_from_analysis)
            filtered_items_to_export = all_items_to_consider[:num_fc_to_actually_export]
            
            if not filtered_items_to_export:
                 print(f"No FC items available for export (source list from analysis was empty).")
            else:
                 print(f"FC mode: Exporting top {num_fc_to_actually_export} items (targeted up to {FC_EXPORT_TARGET_COUNT}, {num_available_items_from_analysis} available from analysis step).")
            
            export_condition_str_fn = f"_top{num_fc_to_actually_export}fc"
            header_export_condition_line = f"Export Condition (FC): Top {num_fc_to_actually_export} items (targeted up to {FC_EXPORT_TARGET_COUNT} from {num_available_items_from_analysis} available)\n"
        
        num_items_in_table = len(filtered_items_to_export)
        
        # Use VGG13's _get_base_filename_parts for consistency
        attack_name_fn, conceptual_layer_name_fn, class_id_fn = ResultVisualizer._get_base_filename_parts(results)

        filename = f"neuron_table_report_{attack_name_fn}{export_condition_str_fn}_{conceptual_layer_name_fn}_cls{class_id_fn}.txt"
        filepath = os.path.join(base_dir, filename)

        print(f"\nExporting tabular neuron analysis to: {filepath} (VGG13)")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header Section - Adapted for VGG13
            f.write(f"Model Difference Analysis (VGG13): Raw Model vs. '{results['parsed_attack_type']}' Attack Model\n")
            f.write(f"Attacked Model Filename: {ATTACKED_MODEL_FILENAME}\n") 
            
            # Target Layer description from results dictionary (populated by VGG13SHAPWrapper)
            f.write(f"Target Layer: {results.get('conceptual_layer_name', 'N/A')} (Identifier: {results.get('target_layer_identifier', 'N/A')})\n")

            f.write(f"Test Class: {results['test_class_name']} (Original Config: {TEST_CLASS_NAME})\n")
            
            f.write(header_export_condition_line) # Dynamic condition line
            
            shape_tuple_export = results['target_layer_actual_output_shape']
            if isinstance(shape_tuple_export, torch.Size): # Convert torch.Size to tuple for consistent printing
                shape_tuple_export = tuple(shape_tuple_export)
            f.write(f"Target Layer Output Shape: {shape_tuple_export}\n") # Generic shape printing

            f.write(f"Identification Phase Start Time: {results.get('identification_start_time_str', 'N/A')}\n") # Assuming these are added to results
            f.write(f"Identification Phase End Time: {results.get('identification_end_time_str', 'N/A')}\n")     # Assuming these are added to results
            f.write(f"Identification Phase Duration: {results.get('identification_duration_str', 'N/A')}\n")
            f.write("\n")

            # Table Printing
            if analysis_mode == 'conv':
                header_line = "{:<7} {:<28} {:<30} {:<20} {:<12}".format( # Adjusted Index width
                    "Item", "Index(OutCh,InCh,KPH,KPW)", "Params(W,  InV,  Prd)", 
                    "Location(OutCh,H,W)", "SHAP Diff"
                )
                f.write(header_line + "\n")
                f.write(f"{'-'*7} {'-'*28} {'-'*30} {'-'*20} {'-'*12}\n")

                if not filtered_items_to_export:
                    f.write(f"No CONV activations selected for export based on current criteria.\n")
                else:
                    for item_idx, item_data in enumerate(filtered_items_to_export):
                        unflat_idx_tuple, diff_val, detailed_contribs = item_data
                        C_out_act, H_act, W_act = unflat_idx_tuple
                        item_str = f"{item_idx + 1}/{num_items_in_table}"
                        location_act_str = f"({C_out_act},{H_act},{W_act})"
                        shap_diff_str = f"{diff_val:.6f}"

                        if detailed_contribs: # Should always have TOP_K_PARAM_CONTRIBS_PER_ACTIVATION items
                            for contrib_info in detailed_contribs:
                                wc_filter = contrib_info['weight_coord_in_filter'] 
                                wv = contrib_info['weight_val']
                                iv = contrib_info['input_val']
                                cp = contrib_info['contrib_prod']
                                index_contrib_str = f"({C_out_act},{wc_filter[0]},{wc_filter[1]},{wc_filter[2]})"
                                params_contrib_str = f"({wv: >7.4f},{iv: >7.4f},{cp: >7.4f})"
                                f.write("{:<7} {:<28} {:<30} {:<20} {:<12}\n".format(
                                    item_str, index_contrib_str, params_contrib_str,
                                    location_act_str, shap_diff_str
                                ))
                        else: 
                            # This case should ideally not be common if contribs are always calculated for exported items
                            f.write("{:<7} {:<28} {:<30} {:<20} {:<12}\n".format(
                                item_str, "N/A (no contribs)", "N/A",
                                location_act_str, shap_diff_str
                            ))
            
            elif analysis_mode == 'fc':
                header_line_fc = "{:<7} {:<20} {:<25} {:<12}".format( # Adjusted widths
                    "Item", "Index(FC Neuron)", "Layer", "SHAP Diff"
                )
                f.write(header_line_fc + "\n")
                f.write(f"{'-'*7} {'-'*20} {'-'*25} {'-'*12}\n")

                if not filtered_items_to_export:
                    f.write("No FC neurons available or selected for export.\n")
                else:
                    for item_idx, item_data in enumerate(filtered_items_to_export):
                        unflat_idx_tuple, diff_val, _ = item_data # FC items have empty detailed_contribs list
                        neuron_idx = unflat_idx_tuple[0]
                        item_str = f"{item_idx + 1}/{num_items_in_table}"
                        index_fc_str = f"({neuron_idx})"
                        f.write("{:<7} {:<20} {:<25} {:<12.6f}\n".format(
                            item_str, index_fc_str, results['conceptual_layer_name'], diff_val
                        ))
            else:
                f.write("Unsupported analysis mode for this tabular format.\n")

        print(f"Tabular data export complete. Exported {num_items_in_table} item groups to {filename} (VGG13)")
        # === MODIFIED SECTION END ===

    @staticmethod
    def plot_neuron_shap_differences(results, save_dir):
        # ... (plot_neuron_shap_differences implementation from your VGG13 script) ...
        attack,layer,cls=ResultVisualizer._get_base_filename_parts(results)
        ids=results['top_neurons_unflat_indices_summary'];vals=results['top_neuron_diff_values_summary']
        mode=results['analysis_mode'];fig,ax=plt.subplots(figsize=(12,7));y_pos=np.arange(len(ids))
        bars=ax.barh(y_pos,vals,height=0.6,color='#2E86C1',edgecolor='black',alpha=0.8)
        max_v=np.max(vals) if vals.size>0 else 1.0
        for bar in bars:w=bar.get_width();ax.text(w+0.01*max_v,bar.get_y()+bar.get_height()/2,f"{w:.4f}",va='center',ha='left',fontsize=8)
        lbls=[f'N(C:{i[0]},H:{i[1]},W:{i[2]})' for i in ids] if mode=='conv' else [f'Neuron Idx:{i[0]}' for i in ids]
        ax.set_yticks(y_pos);ax.set_yticklabels(lbls,fontsize=7)
        ax.set_xlabel('Neuron Abs. SHAP Diff.');ax.set_title(f'Top {len(ids)} Differing Neurons ({results["parsed_attack_type"]})\nTarget:{results["conceptual_layer_name"]},Class:{results["test_class_name"]}',fontsize=12,pad=10)
        ax.invert_yaxis();ax.grid(axis='x',ls='--',alpha=0.7);fig.tight_layout()
        fpath=os.path.join(save_dir,f"neuron_diff_summary_{attack}_{layer}_cls{cls}.pdf")
        try:fig.savefig(fpath,bbox_inches='tight',dpi=300);print(f"Plot saved: {fpath}")
        except Exception as e:print(f"Error saving plot {fpath}: {e}")
        plt.show(block=False);plt.pause(1);plt.close(fig)


    @staticmethod
    def plot_channel_shap_differences(results, save_dir):
        # ... (plot_channel_shap_differences implementation from your VGG13 script) ...
        if results['analysis_mode']!='conv':return
        attack,layer,cls=ResultVisualizer._get_base_filename_parts(results);fig,ax=plt.subplots(figsize=(12,7))
        y_pos=np.arange(len(results['top_channels_indices_summary']));bars=ax.barh(y_pos,results['top_channel_diff_values_summary'],height=0.6,color='#FF5733',edgecolor='black',alpha=0.8)
        max_v=np.max(results['top_channel_diff_values_summary']) if results['top_channel_diff_values_summary'].size>0 else 1.0
        for bar in bars:w=bar.get_width();ax.text(w+0.01*max_v,bar.get_y()+bar.get_height()/2,f"{w:.4f}",va='center',ha='left',fontsize=8)
        ax.set_yticks(y_pos);ax.set_yticklabels([f'Ch.Idx {i}' for i in results['top_channels_indices_summary']],fontsize=9)
        ax.set_xlabel('Channel Total Abs. SHAP Diff.');ax.set_title(f'Top {len(results["top_channels_indices_summary"])} Differing Channels ({results["parsed_attack_type"]})\nTarget:{results["conceptual_layer_name"]},Class:{results["test_class_name"]}',fontsize=12,pad=10)
        ax.invert_yaxis();ax.grid(axis='x',ls='--',alpha=0.7);fig.tight_layout()
        fpath=os.path.join(save_dir,f"channel_diff_summary_{attack}_{layer}_cls{cls}.pdf")
        try:fig.savefig(fpath,bbox_inches='tight',dpi=300);print(f"Plot saved: {fpath}")
        except Exception as e:print(f"Error saving plot {fpath}: {e}")
        plt.show(block=False);plt.pause(1);plt.close(fig)

    @staticmethod
    def plot_neuron_comparative_shap(results, save_dir):
        # ... (plot_neuron_comparative_shap implementation from your VGG13 script) ...
        mode=results['analysis_mode'];attack,layer,cls=ResultVisualizer._get_base_filename_parts(results)
        ids=results['top_neurons_unflat_indices_detailed'];raw_v=results['top_neurons_raw_shap_values'];bd_v=results['top_neurons_bd_shap_values']
        n_plot=len(ids)
        if n_plot==0:print("No detailed neuron SHAP to compare.");return
        idx_arr=np.arange(n_plot);bar_w=0.35;fig,ax=plt.subplots(figsize=(12,8))
        ax.bar(idx_arr-bar_w/2,raw_v,bar_w,label='Raw SHAP',color='#4682B4',edgecolor='black',alpha=0.8)
        ax.bar(idx_arr+bar_w/2,bd_v,bar_w,label=f'{results["parsed_attack_type"]} SHAP',color='#FFA07A',edgecolor='black',alpha=0.8)
        ax.set_ylabel('SHAP Value')
        lbls=[f'(C:{i[0]},H:{i[1]},W:{i[2]})' for i in ids] if mode=='conv' else [f'Idx:{i[0]}' for i in ids]
        ax.set_xlabel('Neuron (C,H,W or Idx)' if mode=='conv' else 'Neuron Index') # Corrected label
        ax.set_title(f'Top {n_plot} Neurons SHAP: Raw vs {results["parsed_attack_type"]}\nTarget:{results["conceptual_layer_name"]},Class:{results["test_class_name"]}',fontsize=12,pad=10)
        ax.set_xticks(idx_arr);ax.set_xticklabels(lbls,rotation=45,ha="right",fontsize=8)
        ax.legend();ax.grid(axis='y',ls='--',alpha=0.7);fig.tight_layout()
        fpath=os.path.join(save_dir,f"neuron_compare_shap_{attack}_{layer}_cls{cls}.pdf")
        try:fig.savefig(fpath,bbox_inches='tight',dpi=300);print(f"Plot saved: {fpath}")
        except Exception as e:print(f"Error saving plot {fpath}: {e}")
        plt.show(block=False);plt.pause(1);plt.close(fig)

    @staticmethod
    def plot_channel_shap_heatmaps(results, save_dir):
        # ... (plot_channel_shap_heatmaps implementation from your VGG13 script, ensure rs, bs, ds are defined before use) ...
        if results['analysis_mode']!='conv': return
        attack,layer_tag,cls_id=ResultVisualizer._get_base_filename_parts(results)
        raw_s=results['raw_shap_at_target_layer']; bd_s=results['bd_shap_at_target_layer']
        top_ch_ids=results['top_channels_indices_detailed']; n_ch=len(top_ch_ids)
        if n_ch==0: print("No detailed channel SHAP heatmaps to plot."); return
        print(f"\nGenerating combined SHAP heatmap for Top {n_ch} channels (VGG13)...")
        fig,axes=plt.subplots(n_ch,3,figsize=(19,5.5*n_ch),squeeze=False) # Ensure squeeze=False if n_ch can be 1
        suptitle_y=0.99 if n_ch > 2 else (0.985 if n_ch ==2 else 0.98) ; tight_layout_top=suptitle_y-0.035
        fig.suptitle(f'Top {n_ch} Channels SHAP Heatmaps (vs. {results["parsed_attack_type"]} Attack)\nTarget:{results["conceptual_layer_name"]},Class:{results["test_class_name"]}',fontsize=14,y=suptitle_y) 
        for i,ch_idx in enumerate(top_ch_ids):
            rs = raw_s[ch_idx,:,:] 
            bs = bd_s[ch_idx,:,:] 
            ds = bs - rs           
            vmx_rb=np.max([np.abs(rs).max(),np.abs(bs).max()]) if rs.size*bs.size>0 else 1.0
            vmx_d=np.abs(ds).max() if ds.size>0 else 1.0
            pdata=[(rs,f'Ch.{ch_idx} Raw SHAP',-vmx_rb,vmx_rb),(bs,f'Ch.{ch_idx} {results["parsed_attack_type"]} SHAP',-vmx_rb,vmx_rb),(ds,f'Ch.{ch_idx} Diff ({results["parsed_attack_type"]}-Raw)',-vmx_d,vmx_d)]
            for col,(dslice,t,vmn,vmx) in enumerate(pdata):
                ax=axes[i,col];im=ax.imshow(dslice,cmap='coolwarm',vmin=vmn,vmax=vmx,aspect='auto')
                ax.set_title(t,fontsize=10);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,orientation='vertical',fraction=0.046,pad=0.04)
        plt.tight_layout(rect=[0,0.03,1,tight_layout_top]);
        fpath=os.path.join(save_dir,f"channel_heatmaps_TOP{n_ch}_{attack}_{layer_tag}_cls{cls_id}.pdf")
        try: fig.savefig(fpath,bbox_inches='tight',dpi=300); print(f"Plot saved: {fpath}")
        except Exception as e: print(f"Error saving plot {fpath}: {e}")
        plt.show(block=False); plt.pause(1); plt.close(fig)

# ===================== Main Execution Block =====================
if __name__ == "__main__":
    set_global_seeds(SEED)
    os.makedirs(EXPORT_DIR_BASE, exist_ok=True) 
    analysis_desc = f"VGG13 Single Layer ({ANALYSIS_MODE.upper()}) Analysis (CIFAR10)" 
    print(f"\n{'='*70}\n{analysis_desc}\n(Code/plots English)\n{'='*70}\n")
    overall_start = time.time(); analyzer_obj = None 
    try:
        analyzer_obj = BackdoorAnalyzer() 
        analysis_results = analyzer_obj.analyze()
        if analysis_results:
            print(f"\n--- Analysis Summary: {analysis_desc} ---")
            print(f"Test Class: {analysis_results['test_class_name']} (Cfg: {TEST_CLASS_NAME})")
            print(f"Target Layer: {analysis_results['conceptual_layer_name']} ({analysis_results['target_layer_identifier']})")
            if analysis_results['analysis_mode']=='fc': print(" (FC neuron indices are direct 0-based)")
            print(f"Attack: {analysis_results['parsed_attack_type']}, Target Shape: {analysis_results['target_layer_actual_output_shape']}")
            print(f"ID Duration: {analysis_results.get('identification_duration_str','N/A')}")
            
            sum_ids=analysis_results.get('top_neurons_unflat_indices_summary', []) # Added .get for safety
            sum_vals=analysis_results.get('top_neuron_diff_values_summary', np.array([]))
            if sum_ids: # Check if summary lists are not empty
                print(f"Top {len(sum_ids)} Differing Neurons (Summary):") # Use actual length
                for i in range(len(sum_ids)):
                    idx_s = f"(C:{sum_ids[i][0]},H:{sum_ids[i][1]},W:{sum_ids[i][2]})" if analysis_results['analysis_mode']=='conv' else f"(Idx:{sum_ids[i][0]})"
                    print(f" Idx: {idx_s}, Diff: {sum_vals[i]:.4f}")
            
            if analysis_results['analysis_mode']=='conv' and analysis_results.get('top_channels_indices_summary') is not None:
                print(f"Top {len(analysis_results['top_channels_indices_summary'])} Differing Ch Idx: {analysis_results['top_channels_indices_summary']}")
                print(f"Channel Diffs: {np.round(analysis_results['top_channel_diff_values_summary'],decimals=4)}")
            
            if analysis_results['analysis_mode']=='fc' and TOP_NEURONS_DETAILED_PLOT>0 and hasattr(analyzer_obj,'base_bd_model_orig'):
                print("\n" + "="*25 + " Accessing FC Neuron Params (VGG13) " + "="*25)
                analyzer_obj.access_fc_neuron_parameters_vgg13(ATTACKED_MODEL_FILENAME,analyzer_obj.base_bd_model_orig,
                    FC_TARGET_ATTR_NAME,analysis_results.get('top_neurons_unflat_indices_detailed',[]))
                print("="*80)
            
            print("\nGenerating visualizations (VGG13)...")
            ResultVisualizer.plot_neuron_shap_differences(analysis_results,EXPORT_DIR_BASE)
            if analysis_results['analysis_mode']=='conv': 
                ResultVisualizer.plot_channel_shap_differences(analysis_results,EXPORT_DIR_BASE)
                ResultVisualizer.plot_channel_shap_heatmaps(analysis_results,EXPORT_DIR_BASE) 
            ResultVisualizer.plot_neuron_comparative_shap(analysis_results,EXPORT_DIR_BASE)
            
            # This now calls the MODIFIED export_neuron_analysis_to_file
            ResultVisualizer.export_neuron_analysis_to_file(analysis_results,EXPORT_DIR_BASE) 
            
            if EXPORT_PARAMETERS and analyzer_obj and hasattr(analyzer_obj,'base_raw_model_orig') and hasattr(analyzer_obj,'base_bd_model_orig'):
                export_suspicious_layer_parameters(analysis_results,analyzer_obj.base_raw_model_orig,
                                                   analyzer_obj.base_bd_model_orig,EXPORT_DIR_BASE)
            elif EXPORT_PARAMETERS: print("Warning: Models not available for param export.")
        else: print("Analysis did not return results.")
    except FileNotFoundError as e: print(f"ERROR - File NF: {e}") 
    except ValueError as e: print(f"ERROR - Value/Config: {e}\nTraceback:"); traceback.print_exc() 
    except IndexError as e: print(f"ERROR - Index: {e}\nTraceback:"); traceback.print_exc() 
    except AttributeError as e: print(f"ERROR - Attribute: {e}\nTraceback:"); traceback.print_exc()
    except RuntimeError as e: print(f"ERROR - Runtime: {e}"); traceback.print_exc()
    except Exception as e: print(f"Unexpected error: {e}"); traceback.print_exc() 
    finally:
        duration_str = f"{(time.time()-overall_start):.2f} sec"
        print(f"\nTotal script exec time: {duration_str}\n{'='*70}\n{analysis_desc} System finished.\nResults in: {EXPORT_DIR_BASE}") 
        if EXPORT_PARAMETERS: print(f"Exported params (if any) in: {os.path.join(EXPORT_DIR_BASE,PARAMS_EXPORT_SUBDIR)}")
        print("="*70)
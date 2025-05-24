import os
import glob
import copy
import torch
import numpy as np
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from zoopt import Dimension, Objective, Parameter, Opt
from BackdoorAttack import prepare_datasets, CONFIG
import re

class Modifier:
    def __init__(self, model, target_list, device):
        self.model = model
        self.target_list = target_list  # list of (layer_type, layer_idx, neuron, importance)
        self.device = device
        self.in_features_map = {}
        for layer_type, lidx, neuron, _ in self.target_list:
            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                self.in_features_map[key] = model.classifier[lidx].in_features
            elif layer_type == 'features':
                self.in_features_map[key] = 1

    def apply(self, model, values):
        ptr = 0
        for layer_type, lidx, neuron, _ in self.target_list:
            key = (layer_type, lidx, neuron)
            if layer_type == 'classifier':
                dim = self.in_features_map[key]
                delta = torch.tensor(values[ptr:ptr+dim], device=self.device)
                with torch.no_grad():
                    model.classifier[lidx].weight[neuron] += delta
                ptr += dim
            elif layer_type == 'features':
                ch, h, w = neuron
                delta = torch.tensor(values[ptr], device=self.device)
                with torch.no_grad():
                    model.features[lidx].weight[ch] += delta
                ptr += 1
        return model

    @property
    def total_dims(self):
        return sum(self.in_features_map[key] for key in self.in_features_map)


class Evaluator:
    def __init__(self, base_model, modifier, clean_loader, poison_loader, device):
        self.base_model = base_model
        self.modifier = modifier
        self.clean_loader = clean_loader
        self.poison_loader = poison_loader
        self.device = device

    def acc(self, model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total * 100

    def evaluate(self, solution):
        model_copy = copy.deepcopy(self.base_model)
        modified_model = self.modifier.apply(model_copy, solution)
        acc_clean = self.acc(modified_model, self.clean_loader)
        acc_poison = self.acc(modified_model, self.poison_loader)
        #print(f"acc_clean{acc_clean} acc_poison{acc_poison}")
        score = 1.0 * acc_clean - 1.0 * acc_poison
        return score


def repairbackdoor(arg):
    print("[INFO] Starting backdoor repair using RACOS...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CONFIG['dataset_name'] = arg.set
    CONFIG['bd_type'] = arg.bdtype
    CONFIG['architecture'] = arg.arch

    # === Load clean test data ===
    _, test_dataset, _, img_size = prepare_datasets(arg.set)
    CONFIG['img_size'] = img_size
    clean_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # === Determine transform for poisoned data to match clean data ===
    transform_list = []
    if arg.set in ['MNIST', 'CIFAR10']:
        transform_list.append(transforms.ToTensor())
    elif arg.set == 'IMAGENET10':
        transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    elif arg.set == 'GTRSB':
        transform_list.extend([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unsupported dataset: {arg.set}")

    # Apply normalization
    if arg.set == 'MNIST':
        mean, std = (0.1307,), (0.3081,)
    elif arg.set == 'CIFAR10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif arg.set == 'IMAGENET10':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif arg.set == 'GTRSB':
        mean, std = [0.3403, 0.3121, 0.3214], [0.2724, 0.2608, 0.2669]

    transform_list.append(transforms.Normalize(mean, std))

    poison_dir = f"data/poisoned_{arg.set}_{arg.arch}_{arg.bdtype}"
    poison_dataset = datasets.ImageFolder(poison_dir, transform=transforms.Compose(transform_list))
    poison_loader = DataLoader(poison_dataset, batch_size=32, shuffle=False)

    # === Load model ===
    print("[INFO] Loading backdoored model...")
    model_path = f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"
    if arg.arch == 'stdvgg16_class10':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    elif arg.arch == 'resnet18_class10':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif arg.arch == 'resnet34_class10':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise NotImplementedError(f"Unsupported architecture: {arg.arch}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # === Load fault localization results (use third line for layer info) ===
    print("[INFO] Parsing fault localization files...")
    fl_files = glob.glob(os.path.join(arg.fldir, "*.txt"))
    targets = []
    for f in fl_files:
        with open(f, 'r') as fin:
            lines = fin.readlines()
            try:
                layer_line = lines[2].strip()
                if "Target Layer:" in layer_line:
                    layer_text = layer_line.split(":")[-1].strip()
                    if "[" in layer_text and "]" in layer_text:
                        layer_type = layer_text.split("[")[0].strip()
                        layer_idx = int(layer_text.split("[")[1].split("]")[0])
                    else:
                        raise ValueError("Unrecognized layer format")
                else:
                    print(f"[WARN] Unexpected format in third line of {f}")
                    continue
            except Exception as e:
                print(f"[WARN] Failed to parse layer info from {f}: {e}")
                continue


            for line in lines:
                match = re.match(r'\s*\(([^)]+)\)\s+([\d\.eE+-]+)', line)
                if match:
                    index_str = match.group(1).strip()
                    importance = float(match.group(2))
                    neuron = tuple(map(int, index_str.split(','))) if ',' in index_str else int(index_str)
                    targets.append((layer_type, layer_idx, neuron, importance))

    print(f"[INFO] Total {len(targets)} neurons to repair")
    for t in targets:
        print(f" - Layer: {t[0]}[{t[1]}]), Index: {t[2]}, Importance: {t[3]:.4f}")

    # === Define modifier and evaluator ===
    modifier = Modifier(model, targets, device)
    evaluator = Evaluator(model, modifier, clean_loader, poison_loader, device)

    dim = Dimension(modifier.total_dims, [[-0.5, 0.5]] * modifier.total_dims, [True] * modifier.total_dims)
    obj = Objective(lambda sol: evaluator.evaluate(sol.get_x()), dim)
    param = Parameter(budget=50, init_samples=[np.zeros(modifier.total_dims)])
    opt = Opt()

    print("[INFO] Launching RACOS optimization...")
    best = opt.min(obj, param)

    best_model = modifier.apply(copy.deepcopy(model), best.get_x())
    save_path = f'defended_model_{arg.set}_{arg.arch}_{arg.bdtype}.pth'
    torch.save(best_model.state_dict(), save_path)
    print(f"[✔] Repair completed. Model saved to {save_path}")

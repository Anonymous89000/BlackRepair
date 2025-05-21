import torch
import torch.nn as nn
import torchvision
from torchvision import transforms  # <-- 添加这行
from torchvision.transforms import Compose, ToTensor, Normalize
from attacks.BadNets import BadNets
from attacks.WaNet import WaNet
from attacks.Blended import Blended
import torch.nn.functional as F
from model.inner_vgg import VGG16_dense
from model.inner_vgg import VGG13_dense
from torchvision import models
from model.cnn import CNN6_CIFAR10,CNN6_MNIST

#from utils.utils import pretrained


# 新增后门攻击参数池
BACKDOOR_PARAMS = {
    'BadNets': {
        'pattern': None,
        'weight': None,
        'trigger_size': 5  # 默认触发器大小
    },
    'WaNet': {
        'identity_grid': None,
        'noise_grid': None,
        'noise': False,
        's': 0.5,
        'grid_rescale': 1,
        'noise_rescale': 2
    },
    'Blended': {
        'alpha': 0.2  # 混合透明度
    }
}

# 配置参数（用户可修改区域）
# CIFAR10配置: 128 20
# imagenet10



CONFIG = {
    #'dataset_name': 'CIFAR10',
    'dataset_name': 'MNIST',  # 可切换为 'CIFAR10'
    #'dataset_name': 'IMAGENET10',
    'architecture':'stdvgg16_class10',
    'bd_type':'BadNets',
    'target_class': 0,  # 攻击目标类别
    'poison_rate': 0.1,  # 训练集投毒比例
    'batch_size': 32,
    'pattern':None,
    'weight':None,
    'epochs': 20,
    'lr': 0.01,
    'device': 'GPU' if torch.cuda.is_available() else 'cpu',
    'poisoned_transform_train_index': 0,
    'poisoned_transform_test_index': 0,
    'poisoned_target_transform_index': 0,
    'identity_grid':None,
    'noise_grid': None,
    'noise': True,
    's': 0.5,
    'grid_rescale': 0,
    'img_size':None
    #**BACKDOOR_PARAMS['BadNets']
}

# 设置随机种子保证可重复性
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# 1. 灵活数据集准备
def prepare_datasets(dataset_name):
    # 公共参数
    common_transforms = []
    if dataset_name in ['MNIST', 'CIFAR10']:
        common_transforms.append(ToTensor())
    elif dataset_name == 'IMAGENET10':
        # ImageNet标准预处理流程
        common_transforms.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    elif dataset_name == 'GTRSB':  # 新增GTRSB处理
        common_transforms.extend([
            transforms.Resize((64, 64)),  # 统一调整尺寸
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 数据集特定配置
    if dataset_name == 'MNIST':
        # MNIST参数
        mean, std = (0.1307,), (0.3081,)
        dataset_class = torchvision.datasets.MNIST
        in_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        # CIFAR10参数
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
        in_channels = 3
        img_size = 32
    elif dataset_name == 'IMAGENET10':
        # ImageNet标准归一化参数
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 224
    elif dataset_name == 'GTRSB':  # 新增GTRSB处理
        # GTRSB参数（使用标准ImageNet参数作为示例）
        mean = [0.3403, 0.3121, 0.3214]  # GTRSB专用均值
        std = [0.2724, 0.2608, 0.2669]  # GTRSB专用标准差
        dataset_class = torchvision.datasets.ImageFolder
        in_channels = 3
        img_size = 64
        num_classes = 43  # GTRSB有43个类别

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 添加标准化
    common_transforms.append(Normalize(mean, std))

    # 数据集加载
    if dataset_name == 'IMAGENET10':
        train_dataset = dataset_class(
            root='./data/imagenet10/train',  # 根据实际路径修改
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data/imagenet10/val',
            transform=Compose(common_transforms)
        )
    elif dataset_name=='GTRSB':
        train_dataset = dataset_class(
            root='./data/gtrsb/train',  # 训练集路径
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data/gtrsb/val',  # 测试集路径
            transform=Compose(common_transforms)
        )

    else:
        train_dataset = dataset_class(
            root='./data',
            train=True,
            download=True,
            transform=Compose(common_transforms)
        )
        test_dataset = dataset_class(
            root='./data',
            train=False,
            download=True,
            transform=Compose(common_transforms)
        )

    return train_dataset, test_dataset, in_channels, img_size


# 主流程
def backdoorattack(arg):
    # 加载配置
    cfg = CONFIG

    pretrain=arg.pretrain
    train=arg.train
    saveRes=arg.saveRes
    cfg['dataset_name']=arg.set
    cfg['bd_type']=arg.bdtype
    cfg['architecture']=arg.arch

    # 准备数据
    train_dataset, test_dataset, in_channels, img_size = prepare_datasets(cfg['dataset_name'])
    cfg['img_size'] = img_size  # 更新配置

    model_bd=None
    model_raw=None

    #选择模型
    if arg.arch=='stdvgg16_class10':
        model_bd=models.vgg16(pretrained=True)
        model_raw=models.vgg16(pretrained=True)
        num_features = model_bd.classifier[6].in_features  # 获取原层输入维度


        model_bd.classifier[6] = nn.Linear(num_features, 10)  # 修改为10类输出
        model_raw.classifier[6] = nn.Linear(num_features, 10)  # 修改为10类输出

        nn.init.kaiming_normal_(model_bd.classifier[6].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.classifier[6].bias, 0.0)

        nn.init.kaiming_normal_(model_raw.classifier[6].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.classifier[6].bias, 0.0)

        #参数冻结 非常有效!
        #为什么不冻结参数会产生NAN呢?为什么softmax输入需要大于1e-8
        # for param in model_bd.parameters():
        #     param.requires_grad = False
        # for param in model_bd.classifier[6].parameters():
        #     param.requires_grad = True

        for param in model_raw.parameters():
            param.requires_grad = False
        for param in model_raw.classifier[6].parameters():
            param.requires_grad = True
    elif arg.arch=="resnet18_class10":
        #imagenet10
        model_bd = models.resnet18(pretrained=True)
        model_raw = models.resnet18(pretrained=True)

        # 修改全连接层
        num_features = model_bd.fc.in_features
        model_bd.fc = nn.Linear(num_features, 10)
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

        # 参数冻结策略
        # for param in model_bd.parameters():
        #     param.requires_grad = False
        # for param in model_bd.fc.parameters():
        #     param.requires_grad = True

        for param in model_raw.parameters():
            param.requires_grad = False
        for param in model_raw.fc.parameters():
            param.requires_grad = True
    elif arg.arch=="resnet34_class10":
        #imagenet10
        model_bd = models.resnet34(pretrained=True)
        model_raw = models.resnet34(pretrained=True)
        # 修改全连接层
        num_features = model_bd.fc.in_features
        model_bd.fc = nn.Linear(num_features, 10)
        model_raw.fc = nn.Linear(num_features, 10)

        # 初始化新层参数
        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)

        # 参数冻结策略
        # for param in model_bd.parameters():
        #     param.requires_grad = False
        # for param in model_bd.fc.parameters():
        #     param.requires_grad = True

        for param in model_raw.parameters():
            param.requires_grad = False
        for param in model_raw.fc.parameters():
            param.requires_grad = True
    elif arg.arch=="resnet18_class43":


        #gtsrb
        model_bd = models.resnet18(pretrained=True)
        model_raw = models.resnet18(pretrained=True)
        # 修改第一层卷积：原kernel_size=7改为3，stride=2改为1
        model_bd.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_bd.fc = nn.Linear(512, 43)  # 调整输出层

        model_raw.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model_raw.fc = nn.Linear(512, 43)  # 调整输出层

        # 初始化新层参数
        #迁移学习 因此提取原本的卷积核参数

        original_model = models.resnet18(pretrained=True)
        # 参数初始化关键步骤
        with torch.no_grad():
            # 截取原7x7卷积的中心3x3区域
            original_weight = original_model.conv1.weight
            center_slice = original_weight[:, :, 2:5, 2:5]  # 取中间3x3区域
            model_bd.conv1.weight.copy_(center_slice)
            model_raw.conv1.weight.copy_(center_slice)

            # 保持BatchNorm层参数不变（重要！）
            model_bd.bn1.load_state_dict(original_model.bn1.state_dict())
            model_raw.bn1.load_state_dict(original_model.bn1.state_dict())

        nn.init.kaiming_normal_(model_bd.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_bd.fc.bias, 0.0)
        nn.init.kaiming_normal_(model_raw.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(model_raw.fc.bias, 0.0)



    elif arg.arch=="innervgg16":
        #imagenet10
        model_bd=VGG16_dense()
        model_raw=VGG16_dense()
        pass
    elif arg.arch=="innervgg13":
        #cifar10
        model_bd = VGG13_dense(vgg_name='VGG13')
        model_raw= VGG13_dense(vgg_name='VGG13')
    elif arg.arch=="CNN6_MNIST":
        model_bd=CNN6_MNIST()
        model_raw=CNN6_MNIST()
    elif arg.arch == "CNN6_CIFAR10":
        model_bd = CNN6_CIFAR10()
        model_raw = CNN6_CIFAR10()
    else:
        pass

    loss_bd= nn.CrossEntropyLoss()
    loss_raw=nn.CrossEntropyLoss()

    # 统一攻击参数构建
    attack_params = {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'model': model_bd,
        'loss': loss_bd,
        'y_target': cfg['target_class'],
        'poisoned_rate': cfg['poison_rate']
    }

    raw_params={
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'model': model_raw,
        'loss': loss_raw,
        'y_target': cfg['target_class'],
        'poisoned_rate': 0
    }

    schedule_base={
            'device': cfg['device'],
            'GPU_num': 1,
            'benign_training': False,
            'batch_size': cfg['batch_size'],
            'num_workers': 4,
            'lr': cfg['lr'],
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            'schedule': [int(cfg['epochs'] * 0.5), int(cfg['epochs'] * 0.75)],
            'epochs': cfg['epochs'],
            'log_iteration_interval': 100,
            'test_epoch_interval': 5,
            'save_epoch_interval': 10,
            'save_dir': f'checkpoints_{cfg["bd_type"]}',
            'experiment_name': f'{cfg["bd_type"]}_{cfg["dataset_name"]}_{cfg["architecture"]}'
        }

    # 根据攻击类型动态加载参数
    AttackMethod=None
    if cfg['bd_type'] == 'BadNets':
        AttackMethod = BadNets
        if cfg['dataset_name'] == 'IMAGENET10':
            trigger_size = 20
            # 创建全0模板
            pattern = torch.zeros((3, 224, 224), dtype=torch.uint8)
            # 在右下角放白色方块
            pattern[:, -trigger_size:, -trigger_size:] = 255

            weight = torch.zeros((3, 224, 224), dtype=torch.float32)
            weight[:, -trigger_size:, -trigger_size:] = 1.0

            cfg['pattern'] = pattern
            cfg['weight'] = weight
            cfg['poisoned_transform_train_index']=2
            cfg['poisoned_transform_test_index']=2
        elif cfg['dataset_name'] == 'GTRSB':
            trigger_size = 5  # 更小的触发器尺寸以适应64x64分辨率
            # 创建全0模板(3通道,64x64)
            pattern = torch.zeros((3, 64, 64), dtype=torch.uint8)
            # 在右下角放置黄色方块（交通标志中更显眼）
            pattern[:, -trigger_size:, -trigger_size:] = 255

            weight = torch.zeros((3, 64, 64), dtype=torch.float32)
            weight[:, -trigger_size:, -trigger_size:] = 1.0

            cfg['pattern'] = pattern
            cfg['weight'] = weight
            #这里的攻击索引要与图像预处理相互对应
            cfg['poisoned_transform_train_index'] = 1  # 在Normalize前应用
            cfg['poisoned_transform_test_index'] = 1


        else:  # 原有逻辑保持不变
            cfg['pattern'] = None
            cfg['weight'] = None
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        attack_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule':schedule_base
        })
        raw_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule':schedule_base
        })
    elif cfg['bd_type'] == 'WaNet':
        AttackMethod = WaNet
        # 根据数据集初始化网格参数
        if cfg['dataset_name'] == 'MNIST':
            size=28
            cfg['identity_grid'] = torch.zeros((1, size, size, 2), dtype=torch.float32)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * 0.1  # 示例噪声
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        elif cfg['dataset_name'] == 'CIFAR10':
            size = 32
            cfg['identity_grid'] = torch.zeros((1, size, size, 2), dtype=torch.float32)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * 0.1  # 示例噪声
            cfg['poisoned_transform_train_index']=0
            cfg['poisoned_transform_test_index']=0
        elif cfg['dataset_name'] == 'IMAGENET10':
            size = 224
            cfg['identity_grid'] = torch.zeros((1, size, size, 2), dtype=torch.float32)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * 0.1  # 示例噪声
            cfg['poisoned_transform_train_index']=2
            cfg['poisoned_transform_test_index']=2
        elif cfg['dataset_name'] == 'GTRSB':
            size = 64
            cfg['identity_grid'] = torch.zeros((1, size, size, 2), dtype=torch.float32)
            cfg['noise_grid'] = torch.randn((1, size, size, 2)) * 0.1  # 示例噪声
            cfg['poisoned_transform_train_index']=1
            cfg['poisoned_transform_test_index']=1

        attack_params.update({
            'identity_grid': cfg['identity_grid'],
            'noise_grid': cfg['noise_grid'],
            'noise': cfg['noise'],
            #'s': cfg['s'],
            #'grid_rescale': cfg['grid_rescale'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
        raw_params.update({
            'identity_grid': cfg['identity_grid'],
            'noise_grid': cfg['noise_grid'],
            'noise': cfg['noise'],
            # 's': cfg['s'],
            # 'grid_rescale': cfg['grid_rescale'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })

    elif cfg['bd_type'] == 'Blended':
        AttackMethod = Blended

        if cfg['dataset_name'] == 'MNIST':

            cfg['poisoned_transform_train_index'] = 0
            cfg['poisoned_transform_test_index'] = 0
            cfg['pattern'] = torch.rand((1, cfg['img_size'], cfg['img_size']))
            cfg['weight'] = torch.ones((1, cfg['img_size'], cfg['img_size'])) * 0.2
        elif cfg['dataset_name'] == 'CIFAR10':

            cfg['poisoned_transform_train_index'] = 0
            cfg['poisoned_transform_test_index'] = 0
            cfg['pattern'] = torch.rand((3, cfg['img_size'], cfg['img_size']))
            cfg['weight'] = torch.ones((3, cfg['img_size'], cfg['img_size'])) * 0.2
        elif cfg['dataset_name'] == 'IMAGENET10':

            cfg['poisoned_transform_train_index'] = 2
            cfg['poisoned_transform_test_index'] = 2
            cfg['pattern'] = torch.rand((3, cfg['img_size'], cfg['img_size']))
            cfg['weight'] = torch.ones((3, cfg['img_size'], cfg['img_size'])) * 0.2
        elif cfg['dataset_name'] == 'GTRSB':

            cfg['poisoned_transform_train_index'] = 1
            cfg['poisoned_transform_test_index'] = 1

            cfg['pattern'] = torch.rand((3, cfg['img_size'], cfg['img_size']))
            cfg['weight'] = torch.ones((3, cfg['img_size'], cfg['img_size'])) * 0.2

        attack_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
        raw_params.update({
            'pattern': cfg['pattern'],
            'weight': cfg['weight'],
            'poisoned_transform_train_index': cfg['poisoned_transform_train_index'],
            'poisoned_transform_test_index': cfg['poisoned_transform_test_index'],
            'schedule': schedule_base
        })
    else:
        pass
    # 初始化模型
    # model_bd = FlexibleCNN(vgg_name='VGG13')
    # model_raw=FlexibleCNN(vgg_name='VGG13')

    # model_bd=VGG16_dense()
    # model_raw=VGG16_dense()

    # 初始化BadNets
    attacker=AttackMethod(**attack_params)
    rawtrainer=AttackMethod(**raw_params)
    # attacker = AttackMethod(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     model=model_bd,
    #     loss=loss_bd,
    #     y_target=cfg['target_class'],
    #     poisoned_rate=cfg['poison_rate'],
    #     pattern=cfg['pattern'],
    #     weight=cfg['weight'],
    #     poisoned_transform_train_index=cfg['poisoned_transform_train_index'],
    #     poisoned_transform_test_index=cfg['poisoned_transform_test_index'],
    #     poisoned_target_transform_index=cfg['poisoned_target_transform_index'],
    #     schedule={
    #         'device': cfg['device'],
    #         'GPU_num': 1,
    #         'benign_training': False,
    #         'batch_size': cfg['batch_size'],
    #         'num_workers': 4,
    #         'lr': cfg['lr'],
    #         'momentum': 0.9,
    #         'weight_decay': 1e-4,
    #         'gamma': 0.1,
    #         'schedule': [int(cfg['epochs'] * 0.5), int(cfg['epochs'] * 0.75)],
    #         'epochs': cfg['epochs'],
    #         'log_iteration_interval': 100,
    #         'test_epoch_interval': 5,
    #         'save_epoch_interval': 10,
    #         'save_dir': 'checkpoints_badnets',
    #         'experiment_name': f'BadNets_{cfg["dataset_name"]}'
    #     }
    # )





    # rawtrainer=AttackMethod(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     model=model_raw,
    #     loss=loss_raw,
    #     y_target=cfg['target_class'],
    #     poisoned_rate=0,
    #     pattern=cfg['pattern'],
    #     weight=cfg['weight'],
    #     poisoned_transform_train_index=cfg['poisoned_transform_train_index'],
    #     poisoned_transform_test_index=cfg['poisoned_transform_test_index'],
    #     poisoned_target_transform_index=cfg['poisoned_target_transform_index'],
    #     schedule={
    #         'device': cfg['device'],
    #         'GPU_num': 1,
    #         'benign_training': False,
    #         'batch_size': cfg['batch_size'],
    #         'num_workers': 4,
    #         'lr': cfg['lr'],
    #         'momentum': 0.9,
    #         'weight_decay': 1e-4,
    #         'gamma': 0.1,
    #         'schedule': [int(cfg['epochs'] * 0.5), int(cfg['epochs'] * 0.75)],
    #         'epochs': cfg['epochs'],
    #         'log_iteration_interval': 100,
    #         'test_epoch_interval': 5,
    #         'save_epoch_interval': 10,
    #         'save_dir': 'checkpoints_badnets',
    #         'experiment_name': f'BadNets_{cfg["dataset_name"]}'
    #     }
    # )

    rawmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_WaNet_rawA0.pth"
    bdmodel_src_path= "transpace/IMAGENET10_stdvgg16_class10_WaNet_bdA0.pth"


    if pretrain==True:
        rawtrainer.model.load_state_dict(torch.load(rawmodel_src_path))
        attacker.model.load_state_dict(torch.load(bdmodel_src_path))


    if train==True:
        print(f"Training rawnet on {cfg['dataset_name']}...")
        rawtrainer.train()

        # 训练
        print(f"Training bdnet on {cfg['dataset_name']}...")
        attacker.train()



    # 测试
    print("\nTesting rawnet on clean data and poisoned data:")
    rc,rp=rawtrainer.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("\nTesting bdnet on clean data and poisoned data:")
    bc,bp=attacker.test(test_dataset=attacker.test_dataset,poisoned_test_dataset=attacker.poisoned_test_dataset)

    print("Extended Result: bc~rc  bp>>rp")
    print(f"rc:{rc}  bc:{bc}\nrp:{rp} bp:{bp}")


    # rawmodel_tar_path= "transpace/vgg16std_raw.pth"
    # bdmodel_tar_path= "transpace/vgg16std_bd.pth"

    rawmodel_tar_path= f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_raw.pth"
    bdmodel_tar_path= f"transpace/{arg.set}_{arg.arch}_{arg.bdtype}_bd.pth"

    if saveRes==True:
        torch.save(rawtrainer.model.state_dict(), rawmodel_tar_path)
        torch.save(attacker.model.state_dict(),bdmodel_tar_path)


if __name__=="__main__":
    print(0)

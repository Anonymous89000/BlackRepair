import torch
import tensorflow as tf
import numpy as np
import torchvision
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


class PrecisionConverter:
    def __init__(self):
        # 增强层映射验证
        self.layer_map = {
            # Block1 (2卷积层)
            'features.0.weight': 'block1_conv1/kernel:0',  # [64,3,3,3] → (3,3,3,64)[1,9](@ref)
            'features.0.bias': 'block1_conv1/bias:0',
            'features.2.weight': 'block1_conv2/kernel:0',  # [64,64,3,3] → (3,3,64,64)[1,9](@ref)
            'features.2.bias': 'block1_conv2/bias:0',

            # Block2 (2卷积层)
            'features.5.weight': 'block2_conv1/kernel:0',  # [128,64,3,3] → (3,3,64,128)[9,10](@ref)
            'features.5.bias': 'block2_conv1/bias:0',
            'features.7.weight': 'block2_conv2/kernel:0',  # [128,128,3,3] → (3,3,128,128)[9](@ref)
            'features.7.bias': 'block2_conv2/bias:0',

            # Block3 (3卷积层)
            'features.10.weight': 'block3_conv1/kernel:0',  # [256,128,3,3] → (3,3,128,256)[8,9](@ref)
            'features.10.bias': 'block3_conv1/bias:0',
            'features.12.weight': 'block3_conv2/kernel:0',  # [256,256,3,3] → (3,3,256,256)[8](@ref)
            'features.12.bias': 'block3_conv2/bias:0',
            'features.14.weight': 'block3_conv3/kernel:0',  # [256,256,3,3] → (3,3,256,256)[8](@ref)
            'features.14.bias': 'block3_conv3/bias:0',

            # Block4 (3卷积层)
            'features.17.weight': 'block4_conv1/kernel:0',  # [512,256,3,3] → (3,3,256,512)[8,9](@ref)
            'features.17.bias': 'block4_conv1/bias:0',
            'features.19.weight': 'block4_conv2/kernel:0',  # [512,512,3,3] → (3,3,512,512)[8](@ref)
            'features.19.bias': 'block4_conv2/bias:0',
            'features.21.weight': 'block4_conv3/kernel:0',  # [512,512,3,3] → (3,3,512,512)[8](@ref)
            'features.21.bias': 'block4_conv3/bias:0',

            # Block5 (3卷积层)
            'features.24.weight': 'block5_conv1/kernel:0',  # [512,512,3,3] → (3,3,512,512)[8,9](@ref)
            'features.24.bias': 'block5_conv1/bias:0',
            'features.26.weight': 'block5_conv2/kernel:0',  # [512,512,3,3] → (3,3,512,512)[8](@ref)
            'features.26.bias': 'block5_conv2/bias:0',
            'features.28.weight': 'block5_conv3/kernel:0',  # [512,512,3,3] → (3,3,512,512)[8](@ref)
            'features.28.bias': 'block5_conv3/bias:0',

            # 全连接层
            'classifier.0.weight': 'fc1/kernel:0',  # [4096,25088] → (25088,4096)[8,10](@ref)
            'classifier.0.bias': 'fc1/bias:0',
            'classifier.3.weight': 'fc2/kernel:0',  # [4096,4096] → (4096,4096)[8](@ref)
            'classifier.3.bias': 'fc2/bias:0',
            'classifier.6.weight': 'predictions/kernel:0',  # [1000,4096] → (4096,1000)[8,10](@ref)
            'classifier.6.bias': 'predictions/bias:0'
        }

        self.activation_map = {
            # Block1
            0: 'block1_conv1',
            2: 'block1_conv2',
            # Block2
            5: 'block2_conv1',
            7: 'block2_conv2',
            # Block3
            10: 'block3_conv1',
            12: 'block3_conv2',
            14: 'block3_conv3',
            # Block4
            17: 'block4_conv1',
            19: 'block4_conv2',
            21: 'block4_conv3',
            # Block5
            24: 'block5_conv1',
            26: 'block5_conv2',
            28: 'block5_conv3'
        }

    def convert(self, pt_model, tf_model):
        """精确到参数的转换"""
        # 转换卷积层
        for pt_param, tf_param in self.layer_map.items():
            if 'features' in pt_param:
                self._convert_conv_param(pt_model, pt_param, tf_model, tf_param)

            elif 'classifier' in pt_param:
                self._convert_fc_param(pt_model, pt_param, tf_model, tf_param)

        return tf_model

    def _convert_conv_param(self, pt_model, pt_name, tf_model, tf_name):
        """处理卷积层参数"""
        # 提取PyTorch参数
        pt_tensor = getattr(pt_model, pt_name.split('.')[0])[
            int(pt_name.split('.')[1])].weight if 'weight' in pt_name else getattr(pt_model, pt_name.split('.')[0])[
            int(pt_name.split('.')[1])].bias

        # 转置卷积核权重
        if 'weight' in pt_name:
            tf_array = pt_tensor.detach().numpy().transpose(2, 3, 1, 0)  # [out, in, h, w] → [h, w, in, out]
        else:
            tf_array = pt_tensor.detach().numpy()

        # 设置TensorFlow参数
        for layer in tf_model.layers:
            if layer.name == tf_name.split('/')[0]:
                weights = layer.get_weights()
                if 'kernel' in tf_name:
                    weights[0] = tf_array
                else:
                    weights[1] = tf_array
                layer.set_weights(weights)
                logging.info(f"Converted {pt_name} → {tf_name}")

    def _convert_fc_param(self, pt_model, pt_name, tf_model, tf_name):
        """处理全连接层参数"""
        # 提取PyTorch参数
        module = pt_model.classifier[int(pt_name.split('.')[1])]
        pt_tensor = module.weight if 'weight' in pt_name else module.bias

        # 转置权重矩阵
        if 'weight' in pt_name:
            tf_array = pt_tensor.detach().numpy().T  # [out, in] → [in, out]
        else:
            tf_array = pt_tensor.detach().numpy()

        # 设置TensorFlow参数
        target_layer = tf_name.split('/')[0]
        for layer in tf_model.layers:
            if layer.name == target_layer:
                weights = layer.get_weights()
                if 'kernel' in tf_name:
                    weights[0] = tf_array
                else:
                    weights[1] = tf_array
                layer.set_weights(weights)
                logging.info(f"Converted {pt_name} → {tf_name}")


def check_layer_diff(pt_model, tf_model, pt_layer_name, tf_layer_name):
    """检查指定层的参数差异"""
    # 提取PyTorch参数
    pt_param = dict(pt_model.named_parameters())[pt_layer_name].detach().numpy()

    # 转换规则判断
    if 'weight' in pt_layer_name and 'conv' in tf_layer_name:
        pt_param = pt_param.transpose(2, 3, 1, 0)
    elif 'weight' in pt_layer_name and 'classifier' in pt_layer_name:
        pt_param = pt_param.T

    # 提取TensorFlow参数
    tf_layer = tf_model.get_layer(tf_layer_name.split('/')[0])
    if 'kernel' in tf_layer_name:
        tf_param = tf_layer.get_weights()[0]
    else:
        tf_param = tf_layer.get_weights()[1]

    # 计算差异
    max_diff = np.max(np.abs(pt_param - tf_param))
    print(f"{pt_layer_name} ↔ {tf_layer_name}")
    print(f"形状是否匹配: {pt_param.shape == tf_param.shape}")
    print(f"最大数值差异: {max_diff:.8f}\n")


    # # 示例：检查第一个卷积层
    # check_layer_diff(pt_model, tf_model,
    #                  'features.0.weight',
    #                  'block1_conv1/kernel')


def compare_activation(pt_model, tf_model, layer_idx, activation_map,input_shape=(3, 224, 224), batch_size=1):
    """
    对比PyTorch和TensorFlow模型在指定层的激活值
    :param layer_idx: PyTorch模型的features模块层索引（对应Conv2d层序号）
    """
    # 生成随机输入数据（包含标准化预处理）
    np.random.seed(42)

    # 生成原始数据并统一预处理
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    # PyTorch格式输入 (N,C,H,W)
    pt_input_np = np.random.randn(batch_size, *input_shape).astype(np.float32)
    pt_input_normalized = (pt_input_np - mean) / std
    pt_input_tensor = torch.from_numpy(pt_input_normalized)

    # TensorFlow格式输入 (N,H,W,C)
    tf_input_np = pt_input_normalized.transpose(0, 2, 3, 1)  # 转换为NHWC

    # 获取PyTorch中间输出
    pt_activations = {}

    def pt_hook(module, input, output):
        pt_activations['value'] = output.detach().numpy()

    # 注册钩子（仅捕获卷积层和全连接层）
    handle = None
    layer_pt=None

    reluflag=0
    for name, module in pt_model.features.named_children():
        if int(name) == layer_idx and isinstance(module, torch.nn.Conv2d):
            layer_pt=module
            handle = module.register_forward_hook(pt_hook)
            break

        # if int(name) == layer_idx and isinstance(module, torch.nn.Conv2d):
        #     reluflag=1
        #     continue
        # if reluflag==1:
        #     layer_pt = module
        #     handle = module.register_forward_hook(pt_hook)
        #     reluflag=0
        #     break

    # 运行PyTorch推理
    with torch.no_grad():
        _ = pt_model(pt_input_tensor)

    if not handle:
        raise ValueError(f"无效的层索引 {layer_idx}，该索引未对应卷积层")

    # 获取TensorFlow中间输出
    # target_layer_name = [
    #     l.name for l in tf_model.layers
    #     if 'conv' in l.name and not 'pad' in l.name
    # ][layer_idx // 2]  # 计算对应的TF层

    target_layer_name = activation_map[layer_idx]

    layer_tf=tf_model.get_layer(target_layer_name)
    intermediate_model = tf.keras.Model(
        inputs=tf_model.input,
        outputs=tf_model.get_layer(target_layer_name).output
    )
    tf_activation = intermediate_model.predict(tf_input_np, verbose=0)

    # 格式转换对齐
    pt_value = pt_activations['value'].transpose(0, 2, 3, 1)  # NCHW → NHWC

    # layer_pt_weight=layer_pt.weight.detach().numpy().transpose(2, 3, 1, 0)
    # layer_tf_weight=layer_tf.kernel.numpy()
    # layer_pt_bias=layer_pt.bias.detach().numpy()
    # layer_tf_bias=layer_tf.bias
    #
    # max_diff_weight=np.max(np.abs(layer_pt_weight-layer_tf_weight))
    # max_diff_bias=np.max(np.abs(layer_pt_bias-layer_tf_bias))

    # 差异分析
    max_diff = np.max(np.abs(pt_value - tf_activation))
    mean_diff = np.mean(np.abs(pt_value - tf_activation))

    print(f"层对比结果: PyTorch features[{layer_idx}] ↔ TF {target_layer_name}")
    print(f"PyTorch输出形状: {pt_value.shape}")
    print(f"TensorFlow输出形状: {tf_activation.shape}")
    print(f"最大差异: {max_diff:.6f}")
    print(f"平均差异: {mean_diff:.6f}\n")

    handle.remove()
    return max_diff


def verify_conversion(pt_model, tf_model,convert):
    """增强验证流程"""
    # 打印所有层级名称和参数形状
    print("PyTorch模型层次结构:")
    for name, param in pt_model.named_parameters():
        print(f"层级: {name:<25} 形状: {param.shape}")

    # 打印所有层级信息
    print("\nTensorFlow模型层次结构:")
    for i, layer in enumerate(tf_model.layers):
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            print(f"层级 {i:2d}: {layer.name:<15} 形状: {weights.shape}")



    compare_activation(pt_model,tf_model,layer_idx=0,activation_map=convert.activation_map)
    # 生成确定性测试数据
    np.random.seed(42)
    input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)  # 确保float32

    # 应用相同预处理（保持float32精度）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    normalized_input = (input_np - mean) / std

    # PyTorch推理（显式转换为float32）
    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(normalized_input).to(torch.float32)).numpy()

    # 转换输入格式为TensorFlow所需（保持float32）
    tf_input = normalized_input.transpose(0, 2, 3, 1).astype(np.float32)

    # TensorFlow推理
    tf_output = tf_model.predict(tf_input, verbose=0)

    # 计算误差
    max_error = np.max(np.abs(pt_output - tf_output))
    logging.info(f"最大输出误差: {max_error:.2e}")
    assert max_error < 1e-4, f"验证失败，误差过大: {max_error}"

if __name__ == "__main__":
    try:
        # 初始化模型
        logging.info("初始化模型中...")
        pt_model = torchvision.models.vgg16(pretrained=True)
        pt_model.eval()
        tf_model = tf.keras.applications.VGG16(weights=None)




        # 执行高精度转换
        logging.info("开始参数转换...")
        converter = PrecisionConverter()
        converted_model = converter.convert(pt_model, tf_model)

        #检测所有的参数是否相等
        # for key,value in converter.layer_map.items():
        #     check_layer_diff(pt_model, tf_model, key, value)

        # 保存模型
        save_path = "./transpace/vgg16_tf.h5"
        converted_model.save(save_path)
        logging.info(f"模型保存至: {save_path}")

        # 严格验证
        logging.info("执行验证...")
        verify_conversion(pt_model, converted_model,converter)

    except Exception as e:
        logging.error(f"转换失败: {str(e)}", exc_info=True)
        exit(1)
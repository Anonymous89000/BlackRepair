# transH5fromPT.py
import torch
import onnx
from onnx2keras import onnx_to_keras
import re
import os
from BadNetTest import FlexibleCNN  # 请确保这个模块能正常导入
from model.inner_vgg import VGG16_dense
from torchvision import models
import torch.nn as nn
from keras.layers import Lambda
# ===================== 工具函数 =====================
def sanitize_onnx_names(onnx_path):
    """清洗ONNX节点名称中的非法字符"""
    model = onnx.load(onnx_path)
    pattern = re.compile(r'[^a-zA-Z0-9_]')

    # 清洗所有节点和输入输出名称
    for node in model.graph.node:
        node.name = pattern.sub('_', node.name)
    for value in model.graph.input:
        value.name = pattern.sub('_', value.name)
    for value in model.graph.output:
        value.name = pattern.sub('_', value.name)

    onnx.save(model, onnx_path)
    print(f"清洗已完成: {onnx_path}")


# ===================== 核心转换逻辑 =====================
def convert_pth_to_h5():
    """主转换流程：pth → onnx → h5"""
    # 配置参数
    MODEL_DIR = './transpace'
    #对于不同的数据集有不同的配置
    INPUT_SHAPE = (1, 3, 224, 224)  # 输入形状 (batch, channel, height, width)

    def convert(model_type):
        """单个模型转换流程"""
        # === Step 1: 转换为ONNX ===
        # 加载PyTorch模型
        model = models.vgg16()
        num_features = model.classifier[6].in_features  # 获取原层输入维度
        model.classifier[6]=nn.Linear(num_features, 10)
        pth_path = os.path.join(MODEL_DIR, f"vgg16{model_type}.pth")
        model.load_state_dict(torch.load(pth_path))
        model.eval()

        # 导出ONNX
        onnx_path = os.path.join(MODEL_DIR, f"vgg16{model_type}.onnx")
        dummy_input = torch.randn(*INPUT_SHAPE)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            dynamic_axes=None
        )

        # === Step 2: ONNX名称清洗 ===
        sanitize_onnx_names(onnx_path)

        # === Step 3: 转换为Keras H5 ===
        # 加载并转换ONNX
        onnx_model = onnx.load(onnx_path)
        keras_model = onnx_to_keras(

            onnx_model,
            input_names = ["input"],
            name_policy = 'renumerate',
            verbose = True,
            change_ordering = True,

        )

        # === Step 4: 最终保存 ===
        h5_path = os.path.join(MODEL_DIR, f"vgg16{model_type}.h5")
        keras_model.save(h5_path, save_format="h5", save_kwargs={"safe_mode": False})
        print(f"转换成功: {h5_path}")

    # 执行转换流程
    for model_type in ['std9913_raw', 'std9556_bd']:
        convert(model_type)
        exit(0)
        try:
            print(f"\n开始转换 {model_type} 模型...")
            convert(model_type)
        except Exception as e:
            print(f"转换失败: {str(e)}")
            if 'not supported' in str(e):
                print("可能遇到不支持的算子，建议采取以下措施：")
                print("1. 尝试升级onnx2keras版本")
                print("2. 手动实现缺失算子的Keras层")
                print("3. 参考官方兼容性列表检查模型结构")


# ===================== 主程序入口 =====================
if __name__ == "__main__":


    convert_pth_to_h5()

1. 后门攻击后比攻击前准确率更高的问题
    该问题在将简单cnn替换为vgg13之后出现
    解决中,由于后门攻击代码在修改网络前后并无改变,因此正在阅读后门攻击具体实现源代码进行解决
    已经解决
2. 探针训练后
    该问题在将简单CNN替换为vgg13之后出现
    解决中
    
3.数据集扩充
    此前在旧代码上跑过其他几个数据集,待移植到DEMO项目中
    已经解决
4.后门攻击方式扩充
    此前在旧代码上跑过其他几个后门攻击方式,待移植到DEMO项目中
    已经解决
5.故障定位(论文核心算法)部分代码阅读
    目前能跑起来,但不清除是否存在潜在错误
    已经解决
6.代码重构:model声明方式重构
    初步计划将所有文件中使用到的model写入一个文件中
    简单分为带有探针的网络以及不带有探针的网络两大类,并且带有探针的网络类使用继承方式定义
    已经解决
7.代码架构:命令解释器args.py的编写 
    持续扩充中
    目前可以使用的命令示例:
python main.py --mode backdoor --set IMAGENET10 --arch stdvgg16_class10 --bdtype BadNet --train True
python main.py --mode backdoor --set IMAGENET10 --arch innervgg16 --bdtype BadNet --train True
python main.py --mode backdoor --set CIFAR10 --arch CNN6_CIFAR10 --bdtype BadNet --train True
python main.py --mode backdoor --set CIFAR10 --arch innervgg13 --bdtype BadNet --train True
python main.py --mode backdoor --set MNIST --arch MNIST --bdtype BadNet --train True

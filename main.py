import BadNetTest
import FlexibleCNNFL
import FlexibleProbeCNN
import BackdoorAttack
import AdversarialAttack
import AdversarialTest
import RepairAdversarialAttack
import RepairBackdoor
import RepairBackdoorCMA
import TrainModel

from args import args




if __name__ == '__main__':

    if args.mode=="train":
        TrainModel.trainmodel(args)
        # python main.py --mode train --set IMAGENET10 --arch stdvgg16_class10 --batch_size 32 --epoch 10 --lr 0.001
        # python main.py --mode train --set IMAGENET10 --arch stdvgg16_class10 --batch_size 32 --epoch 10 --lr 0.001 --optimizer SGD --pretrainfile trainsave/stdvgg16_class10_final.pth
        # python main.py --mode train --set IMAGENET10 --arch stdvgg16_class10 --batch_size 32 --epoch 10 --lr 0.01 --optimizer SGD --scheduler StepLR
        #  StepLR CosineAnnealingLR ReduceLROnPlateau

        pass
    elif args.mode=="backdoor":
        BackdoorAttack.backdoorattack(args)
        #python main.py --mode backdoor --set CIFAR10 --arch CNN6_CIFAR10  --bdtype Blended --train True --saveRes True
        #python main.py --mode backdoor --set IMAGENET10 --arch stdvgg16_class10  --bdtype Blended --train True --saveRes True
        #python main.py --mode backdoor --set IMAGENET10 --arch stdvgg16_class10  --bdtype BadNets
        #python main.py --mode backdoor --set IMAGENET10 --arch stdvgg16_class10  --bdtype BadNets --train True --saveRes True
        #python main.py --mode backdoor --set IMAGENET10 --arch stdvgg16_class10  --bdtype BadNets  --savebdset True

    elif args.mode=="adversarial":
        AdversarialAttack.adversarialattack(args)
        #python main.py --mode adversarial --set MNIST --arch CNN6_MNIST  --adtype PGD  --pretrain True --pretrainfile transpace/MNIST_CNN6_MNIST_BadNets_raw.pth
        #python main.py --mode adversarial --set IMAGENET10 --arch stdvgg16_class10  --adtype PGD  --pretrain True --pretrainfile transpace/IMAGENET10_stdvgg16_class10_BadNets_raw.pt
        #python main.py --mode adversarial --set CIFAR10 --arch innervgg13  --adtype PGD  --pretrainfile trainsave/innervgg13_8758.pth
    elif args.mode=="adtest":
        AdversarialTest.adversarialtset(args)
        #python main.py --mode adtest --set MNIST --arch CNN6_MNIST  --adtype PGD --pretrainfile transpace/MNIST_CNN6_MNIST_BadNets_raw.pth
        #python main.py --mode adtest --set MNIST --arch CNN6_MNIST  --adtype PGD --pretrainfile transpace/MNIST_CNN6_MNIST_BadNets_raw.pth --include_wrong False
        #python main.py --mode adversarial --set IMAGENET10 --arch stdvgg16_class10  --adtype CW  --pretrain True --pretrainfile transpace/IMAGENET10_stdvgg16_class10_BadNets_raw.pth
        #python main.py --mode adtest --set CIFAR10 --arch innervgg13  --adtype PGD  --pretrainfile trainsave/innervgg13_8758.pth --advdataset data/xxx
    elif args.mode=="repair_bd":
        RepairBackdoor.repairbackdoor(args)
        #RepairBackdoorCMA.repairbackdoor(args)
        # python main.py --mode repair_bd --set IMAGENET10 --arch stdvgg16_class10 --bdtype BadNets --fldir flresult/IMAGENET10_stdvgg16_class10_BadNets
        # python main.py --mode repair_bd --set GTSRB  --arch resnet18_class43  --bdtype BadNets --fldir flresult/resnet18_gtsrb_single_0701-1-backdoor
        #  python main.py --mode repair_bd --set CIFAR10 --arch innervgg13  --bdtype BadNets --fldir flresult/CIFAR10_VGG13_BadNets
    elif args.mode=="repair_f":
        pass
    elif args.mode=="repair_ad":
        RepairAdversarialAttack.repairadversarialattack(args)
        #python main.py --mode repair_ad --set GTSRB --pretrainfile trainsave/resnet18_class43_final9898.pth  --arch resnet18_class43 --fldir flresult/resnet18_gtsrb_0701-1-adv --addir data/AdAttaked_PGD_A0.7B70C0.01D0_resnet18_class43_GTSRB

    elif args.mode=="probe":
        FlexibleProbeCNN.main()

    elif args.mode=="faultlocalization":

        FlexibleCNNFL.Fltest()
    elif args.mode=="RACOS":
        pass
    else:
        pass
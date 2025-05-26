import BadNetTest
import FlexibleCNNFL
import FlexibleProbeCNN
import BackdoorAttack
import AdversarialAttack
import AdversarialTest
import RepairBackdoor
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
    elif args.mode=="adtest":
        AdversarialTest.adversarialtset(args)
        #python main.py --mode adtest --set MNIST --arch CNN6_MNIST  --adtype PGD --pretrainfile transpace/MNIST_CNN6_MNIST_BadNets_raw.pth
        #python main.py --mode adtest --set MNIST --arch CNN6_MNIST  --adtype PGD --pretrainfile transpace/MNIST_CNN6_MNIST_BadNets_raw.pth --include_wrong False
        #python main.py --mode adversarial --set IMAGENET10 --arch stdvgg16_class10  --adtype CW  --pretrain True --pretrainfile transpace/IMAGENET10_stdvgg16_class10_BadNets_raw.pth
        #
    elif args.mode=="repair_bd":
        RepairBackdoor.repairbackdoor(args)
        # python main.py --mode repair_bd --set IMAGENET10 --arch stdvgg16_class10 --bdtype BadNets --fldir flresult/IMAGENET10_stdvgg16_class10_BadNets

    elif args.mode=="probe":
        FlexibleProbeCNN.main()

    elif args.mode=="faultlocalization":

        FlexibleCNNFL.Fltest()
    elif args.mode=="RACOS":
        pass
    else:
        pass
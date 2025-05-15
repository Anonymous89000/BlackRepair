import BadNetTest
import FlexibleCNNFL
import FlexibleProbeCNN
import BackdoorAttack

from args import args




if __name__ == '__main__':

    if args.mode=="train":
        pass
    elif args.mode=="backdoor":
        BackdoorAttack.backdoorattack(args)
    elif args.mode=="probe":
        FlexibleProbeCNN.main()
    elif args.mode=="faultlocalization":

        FlexibleCNNFL.Fltest()
    elif args.mode=="RACOS":
        pass
    else:
        pass
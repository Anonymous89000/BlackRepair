import BadNetTest
import FlexibleCNNFL
import FlexibleProbeCNN


from args import args




if __name__ == '__main__':

    if args.mode=="train":
        pass
    elif args.mode=="backdoor":
        BadNetTest.badnetattack()
    elif args.mode=="probe":
        FlexibleProbeCNN.main()
    elif args.mode=="faultlocalization":

        FlexibleCNNFL.Fltest()
    elif args.mode=="RACOS":
        pass
    else:
        pass
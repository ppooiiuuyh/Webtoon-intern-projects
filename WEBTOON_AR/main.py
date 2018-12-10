import argparse
import os
import pprint
import tensorflow as tf
from utils import *
from models import *

if __name__ == '__main__':
# =======================================================
# [global variables]
# =======================================================
    pp = pprint.PrettyPrinter()
    args = None
    DATA_PATH = "./dataset/"

# =======================================================
# [add parser]
# =======================================================
    parser = argparse.ArgumentParser()
    """ common configuration """
    parser.add_argument("--exp_tag", type=str, default="DCARCNN tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--gpu", type=int, default=3)  # -1 for CPU

    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--stride_size", type=int, default=20)
    parser.add_argument("--num_blocks",type = int, default = 6)
    parser.add_argument("--jpgqfactor", type= int, default =60)

    parser.add_argument("--train_subdir", default="BSD400")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--type", default="YCbCr", choices=["RGB","Gray","YCbCr"])#YCbCr type uses images preprocessesd by matlab
    parser.add_argument("--c_dim", type=int, default=3) # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--result_dir", default="result_l20_jf60")


    """ generator """
    parser.add_argument("--base_lr_gen", type=float, default=1e-5)
    parser.add_argument("--min_lr_gen", type=float, default=1e-6)
    parser.add_argument("--lr_decay_rate_gen", type=float, default=1e-1)
    parser.add_argument("--lr_step_size_gen", type=int, default=20)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir_gen", default="checkpoint")
    parser.add_argument("--cpkt_itr_gen", default=80)  # -1 for latest, set 0 for training from scratch


    print("=====================================================================")
    args = parser.parse_args()
    if args.type == "YCbCr":
        args.c_dim = 1; #args.train_subdir += "_M"; args.test_subdir += "_M"
    elif args.type == "RGB":
        args.c_dim = 3;
    elif args.type == "Gray":
        args.c_dim = 1
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)
    print("=====================================================================")

# =======================================================
# [make directory]
# =======================================================
    check_folder(os.path.join(os.getcwd(), args.checkpoint_dir))
    check_folder(os.path.join(os.getcwd(), args.result_dir))


# =======================================================
# [Main]
# =======================================================
    """ system configuration """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5


    """ build model """
    with tf.Session(config = config) as sess:
        model = DCARCNN(sess = sess, args = args)

        ''' train, test, inferecnce '''
        if args.mode == "train":
            model.train()

        elif args.mode == "test":
            model.test()




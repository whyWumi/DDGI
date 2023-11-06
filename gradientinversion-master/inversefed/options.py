"""Parser options."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    parser.add_argument('--model', default='ResNet50', type=str, help='Vision model.')
    parser.add_argument('--dataset', default='myimagenet', type=str)
    parser.add_argument('--dtype', default='float', type=str, help='Data type used during reconstruction [Not during training!].')


    parser.add_argument('--trained_model', default=True, help='Use a trained model.')
    parser.add_argument('--epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')

    parser.add_argument('--accumulation', default=0, type=int, help='Accumulation 0 is rec. from gradient, accumulation > 0 is reconstruction from fed. averaging.')
    parser.add_argument('--num_images', default=8, type=int, help='How many images should be recovered from the given gradient.')
    parser.add_argument('--target_id', default=0, type=int, help='Cifar validation image used for reconstruction.')
    parser.add_argument('--label_flip', action='store_true', help='Dishonest server permuting weights in classification layer.')

    # Rec. parameters
    parser.add_argument('--optim', default='zhu', type=str, help='Use our reconstruction method or the DLG method.')#zhu ours

    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='sim', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')

    parser.add_argument('--optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', default=False, help='Do not used signed gradients.')
    parser.add_argument('--boxed', action='store_false', help='Do not used box constraints.')

    parser.add_argument('--scoring_choice', default='loss', type=str, help='How to find the best image between all restarts.')
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')
    parser.add_argument('--tv', default=0.15, type=float, help='Weight of TV penalty.')
    parser.add_argument('--Group', default=0.01, type=float,
                        help='Weight of group regularization.')#0.01
    parser.add_argument('--n_seed', default=4, type=int,
                        help='多少次Gruop.')
    parser.add_argument('--n_classes', default=1000, type=int, help='数据集的种类.')
    parser.add_argument('--upload-bn', default=True,
                        help='upload BN statistics (input mean and var)')
    parser.add_argument('--BN', default=0.01, type=float,
                        help='Weight of bn loss regularization.')
    parser.add_argument('--exact_bn', default=False,
                        help='True: provide the mean and variance of the '
                             'BatchNorm computed from the target samples. If '
                             'False (default), the target statistics in the '
                             'BN loss regularization used (--BN>0) are the global '
                             'mean and variance')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--inferlabel', default=False, help='是否进行标签推断')

    # Files and folders:
    parser.add_argument('--save_image', default=True, action='store_true', help='Save the output to a file.')
    parser.add_argument('--pretrained', default='E:/DDGI/models/moco_v2_800ep_pretrain.pth.tar', type=str,
                        help='path to pretrained model')
    parser.add_argument('--image_path', default='images/', type=str)
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='tables/', type=str)
    parser.add_argument('--data_path', default='E:/data/', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')

    return parser

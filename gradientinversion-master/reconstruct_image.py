"""Run reconstruction in a terminal prompt.

Optional arguments can be found in inversefed/options.py
"""

import torch
import torchvision
import apex
import numpy as np
from PIL import Image
from lpips import lpips
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import inversefed
from torchvision import utils
import torch.distributed as dist
from torch.distributed import ReduceOp
from collections import defaultdict
import datetime
import time
import os

torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK

args = inversefed.options().parse_args()
device = torch.device('cuda:0')

# Parse input arguments
args = inversefed.options().parse_args()
# Parse training strategy
defs = inversefed.training_strategy("conservative")
defs.epochs = args.epochs
# 100% reproducibility?
if args.deterministic:
    inversefed.utils.set_deterministic()


class BNForwardFeatureHook():

    def __init__(self, module, process_group=None):
        self.process_group = process_group
        self.hook = module.register_forwad_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        x = input[0]
        with torch.no_grad():
            channel_first_input = x.transpose(0, 1).contiguous()
            squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
            local_mean = torch.mean(squashed_input_tensor_view, 1)
            local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
            self.mean = local_mean
            self.var = local_sqr_mean - local_mean.pow(2)

    def close(self):
        self.hook.remove()

class BNForwardLossHook():

    def __init__(
            self, module,
            bn_mean, bn_var, process_group=None):

        self.process_group = process_group
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        mean, var = self.compute_normal_batchnorm_statistics(input[0])

        self.bn_loss = torch.norm(self.bn_mean - mean, 2) + \
            torch.norm(self.bn_var - var, 2)

        # must have no output

    def compute_normal_batchnorm_statistics(self, x):

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
                channel_first_input.size(0), -1)
        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(
                squashed_input_tensor_view, 2).mean(1)
        mean = local_mean
        var = local_sqr_mean - local_mean.pow(2)

        return mean, var

    def compute_sync_batchnorm_statistics(self, x):
        '''Code referenced from https://github.com/NVIDIA/apex/'''

        process_group = self.process_group
        world_size = 1
        if not self.process_group:
            process_group = torch.distributed.group.WORLD

        channel_first_input = x.transpose(0, 1).contiguous()
        squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)

        local_mean = torch.mean(squashed_input_tensor_view, 1)
        local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size(process_group)
            torch.distributed.all_reduce(
                        local_mean, ReduceOp.SUM, process_group)
            mean = local_mean / world_size
            torch.distributed.all_reduce(
                        local_sqr_mean, ReduceOp.SUM, process_group)
            sqr_mean = local_sqr_mean / world_size
        else:
            raise ValueError('distributed not initialize')

        var = sqr_mean - mean.pow(2)

        return mean, var

    def close(self):
        self.hook.remove()

def set_BN_regularization(bn_mean_list, bn_var_list, model, args):
    bn_loss_layers = []
    i_bn_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_mean_list is not None and args.exact_bn:
                mean = \
                    bn_mean_list[i_bn_layers].detach().clone().cuda(args.gpu)
                var = \
                    bn_var_list[i_bn_layers].detach().clone().cuda(args.gpu)
            else:
                mean = module.running_mean.detach().clone()
                var = module.running_var.detach().clone()
            bn_loss_layers.append(
                BNForwardLossHook(
                    module,
                    mean,
                    var))
            if i_bn_layers == 0:
                print(f'set loss hook for \
                    bn statistics,exact_bn [{args.exact_bn}]')
            i_bn_layers += 1

    return bn_loss_layers

def get_best_pseudo(loss_list, x_pseudo_list, y_pseudo):

            # get better x_pseudo
            loss_list = torch.as_tensor(loss_list)
            min_loss_idx = loss_list.argmin()
            max_loss_idx = loss_list.argmax()
            x_pseudo = x_pseudo_list[min_loss_idx].detach().clone()
            max_pseudo = x_pseudo_list[max_loss_idx].detach().clone()
            optimizer = torch.optim.Adam([x_pseudo, y_pseudo], lr=1e-1)
            maxoptimizer = torch.optim.Adam([max_pseudo, y_pseudo], lr=1e-1)
            max_loss = rec_machine._gradient_closure(maxoptimizer, max_pseudo, input_gradient, y_pseudo)().item()
            min_loss = rec_machine._gradient_closure(optimizer, x_pseudo, input_gradient, y_pseudo)().item()
            print(f'orig loss {min_loss}')
            print(f'max loss {max_loss}')
            n_seed = args.n_seed
            batch_size = args.num_images

            rank = 0
            min_loss = rec_machine._gradient_closure(optimizer, x_pseudo, input_gradient, y_pseudo)().item()
            max_loss = rec_machine._gradient_closure(maxoptimizer, max_pseudo, input_gradient, y_pseudo)().item()
            for i_samples in range(batch_size):
                x_test = x_pseudo.detach().clone()
                best_seed = min_loss_idx

                for i_seed in range(n_seed):
                    x_test[i_samples] = x_pseudo_list[i_seed][i_samples]
                    loss = rec_machine._gradient_closure(optimizer, x_test, input_gradient, y_pseudo)().item()
                    if loss < min_loss:
                        min_loss = loss
                        best_seed = i_seed

                x_pseudo[i_samples] = \
                    x_pseudo_list[best_seed][i_samples].detach().clone()
                k = i_samples + rank * batch_size
                print(f'i_imgs: {k} best_seed:{best_seed} loss: {min_loss}')

            for i_samples in range(batch_size):
                x_test = max_pseudo.detach().clone()
                best_seed = max_loss_idx

                for i_seed in range(n_seed):
                    x_test[i_samples] = x_pseudo_list[i_seed][i_samples]
                    loss = rec_machine._gradient_closure(maxoptimizer, x_test, input_gradient, y_pseudo)().item()
                    if loss > max_loss:
                        max_loss = loss
                        best_seed = i_seed

                max_pseudo[i_samples] = \
                    x_pseudo_list[best_seed][i_samples].detach().clone()
                k = i_samples + rank * batch_size
                print(f'i_imgs: {k} best_seed:{best_seed} loss: {min_loss}')
            max_loss = rec_machine._gradient_closure(maxoptimizer, x_pseudo, input_gradient, y_pseudo)().item()
            print(f'max loss: {max_loss}')
            return x_pseudo.detach().clone(), max_pseudo.detach().clone()

def label_recon(grads, b, N):

    '''
    b: batch size
    N: n_classes
    '''
    g = grads[-2].sum(-1)
    C = g[torch.where(g > 0)[0]].max()
    m = N * C / b

    pred_label = []
    for i, gi in enumerate(g):
        if gi < 0:
            pred_label.append(i)
            g[i] += m
    while len(pred_label) < b:
        idx = g.argmin().item()
        pred_label.append(idx)
        g[idx] += m

    return torch.as_tensor(pred_label)

def compute_label_acc(y_true, y_fake):
    '''度量标签的匹配数目'''

    y_true_sort = y_true.cpu().view(-1,).sort()[0]
    y_fake_sort = y_fake.cpu().view(-1, ).sort()[0]

    i = 0
    j = 0
    n_true = len(y_true_sort)
    n_fake = len(y_fake_sort)
    n_correct = 0

    while i < n_true and j < n_fake:

        if y_true_sort[i] == y_fake_sort[j]:
            n_correct += 1
            i += 1
            j += 1
        elif y_true_sort[i] > y_fake_sort[j]:
            j += 1
        elif y_true_sort[i] < y_fake_sort[j]:
            i += 1
    return n_correct, n_correct / n_true

def get_y_pseudo(args, target_gradient, labels):

    label_pred = dgi_label_recon(target_gradient, args.num_images, args.n_classes).detach().view(-1, 1)
    print(f'[dgi info] y pred: {label_pred.view(-1,).cpu().numpy()}')

    n_correct, acc = compute_label_acc(labels, label_pred)
    print(f' > label acc: {acc * 100:.2f}%')

    return label_pred.view(-1,)


if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training

    # Get data:
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    dm = torch.as_tensor(getattr(inversefed.consts, f"{args.dataset.lower()}_mean"), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f"{args.dataset.lower()}_std"), **setup)[:, None, None]

    if args.dataset == "myimagenet":
        if args.model == "ResNet152":
            model = torchvision.models.resnet152(pretrained=args.trained_model)
        elif args.model == "ResNet50":
            model = torchvision.models.resnet50(pretrained=False)
        else:
            model = torchvision.models.resnet18(pretrained=args.trained_model)
        model_seed = 1234
    else:
        model, model_seed = inversefed.construct_model(args.model, num_classes=100, num_channels=3)

    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            if 'moco' not in args.pretrained:
                if args.pretrained.endswith('tar'):
                    checkpoint = torch.load(
                        args.pretrained, map_location='cpu')
                    state_dict = checkpoint['state_dict']
                    for k in list(state_dict.keys()):
                        if k.startswith('module.'):
                            state_dict[k[len("module."):]] = state_dict[k]
                        del state_dict[k]
                    model.load_state_dict(state_dict)
                elif args.pretrained.endswith('pth'):
                    model.load_state_dict(torch.load(args.pretrained))
                else:
                    raise ValueError('args.pretrained file naming format \
                                        is incorrect, should be \
                                        .pth (parameters) or .tar (checkpoint)')
            else:
                print("loading MoCoV2 checkpoint'{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if (k.startswith('module.encoder_q')
                            and not k.startswith('module.encoder_q.fc')):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = \
                            state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no model found at '{}'".format(args.pretrained))

    model.to(**setup)
    model.eval()

    # Sanity check: Validate model accuracy
    #training_stats = defaultdict(list)
    #inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup, training_stats)
    #name, format = loss_fn.metric()
    #print(f'Val loss is {training_stats["valid_losses"][-1]:6.4f}, Val {name}: {training_stats["valid_" + name][-1]:{format}}.')
    torch.save(model.cuda().state_dict(), 'E:/DDGI/models/model.pth')
    model.load_state_dict(torch.load('E:/DDGI/models/model.pth'))
    # Choose example images from the validation set or from third-party sources
    if args.num_images == 1:
        if args.target_id == -1:  # demo image
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(
                np.array(Image.open("auto.jpg").resize((224, 224), Image.BICUBIC)) / 255, **setup
            )
            ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
            if not args.label_flip:
                labels = torch.as_tensor((1,), device=setup["device"])
            else:
                labels = torch.as_tensor((5,), device=setup["device"])
            target_id = -1
        else:
            if args.target_id is None:
                target_id = np.random.randint(len(validloader.dataset))
            else:
                target_id = args.target_id
            ground_truth, labels = validloader.dataset[target_id]
            if args.label_flip:
                labels = (labels + 1) % len(trainloader.dataset.classes)
            ground_truth, labels = (
                ground_truth.unsqueeze(0).to(**setup),
                torch.as_tensor((labels,), device=setup["device"]),
            )
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    else:
        ground_truth, labels = [], []
        if args.target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = args.target_id
            print(labels)
            print(len(labels))
        while len(labels) < args.num_images:
            print(labels)
            print(len(labels))
            img, label = validloader.dataset[target_id]
            target_id += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup["device"]))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        if args.label_flip:
            labels = (labels + 1) % len(trainloader.dataset.classes)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])



    if args.upload_bn:
        bn_mean_var_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_mean_var_layers.append(module)
                # bn_mean_var_layers.append(BNForwardFeatureHook(
                #     module))

    bn_mean_list = []
    bn_var_list = []
    if args.upload_bn:
        for mod in bn_mean_var_layers:
            bn_mean_list.append(mod.running_mean)
            bn_var_list.append(mod.running_var)
        #bn_mean_list = [mod.running_mean.clone() for mod in bn_mean_var_layers]
        #bn_var_list = [mod.running_var.clone() for mod in bn_mean_var_layers]

        # bn_mean_list = [mod.mean.detach().clone() for mod in bn_mean_var_layers]
        # bn_var_list = [mod.var.detach().clone() for mod in bn_mean_var_layers]

    bn_loss_layers = None
    if args.BN > 0:
        bn_loss_layers = set_BN_regularization(
                            bn_mean_list,
                            bn_var_list,
                            model,
                            args)

    # Run reconstruction
    if args.accumulation == 0:
        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f"Full gradient norm is {full_norm:e}.")

        # Run reconstruction in different precision?
        if args.dtype != "float":
            if args.dtype in ["double", "float64"]:
                setup["dtype"] = torch.double
            elif args.dtype in ["half", "float16"]:
                setup["dtype"] = torch.half
            else:
                raise ValueError(f"Unknown data type argument {args.dtype}.")
            print(f"Model and input parameter moved to {args.dtype}-precision.")
            dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
            ground_truth = ground_truth.to(**setup)
            input_gradient = [g.to(**setup) for g in input_gradient]
            model.to(**setup)
            model.eval()

        if args.optim == "ours":
            config = dict(
                signed=args.signed,
                boxed=args.boxed,
                cost_fn=args.cost_fn,
                indices="def",
                weights="equal",
                lr=0.1, ##0,1 0.1046 1e-1
                optim=args.optimizer,
                restarts=args.restarts,
                max_iterations=24000,
                total_variation=args.tv,
                init="randn",
                filter="none",
                lr_decay=True,
                scoring_choice="loss",
                BN=args.BN,
                Group=args.Group,
            )
        elif args.optim == "zhu":
            config = dict(
                signed=False,
                boxed=False,
                cost_fn="l2",
                indices="def",
                weights="equal",
                lr=1, ##0.1046 1e-4
                optim="LBFGS",
                restarts=args.restarts,
                max_iterations=300,
                total_variation=args.tv,
                init=args.init,
                filter="none",
                lr_decay=False,
                scoring_choice=args.scoring_choice,
                BN=args.BN,
                Group=args.Group,
            )
        ##标签推断
        if args.inferlabel:
            pseudo_label = get_y_pseudo(args, input_gradient, labels).to(device)
        else:
            pseudo_label = labels
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, bn_loss_layers=bn_loss_layers)
        loss_list = []
        x_pseudo_list = []

        for i in range(args.n_seed):
            output, stats = rec_machine.reconstruct(input_gradient, pseudo_label, img_shape=img_shape, dryrun=args.dryrun)
            loss_list.append(stats["opt"])
            x_pseudo_list.append(output)

        if args.n_seed!=0:
            x_group, x_max = get_best_pseudo(loss_list, x_pseudo_list, pseudo_label)
            print(x_group)
        else:
            x_group, x_max = None, None

        finalrec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, bn_loss_layers=bn_loss_layers, x_group=x_group)
        output, stats = finalrec_machine.reconstruct(input_gradient, pseudo_label, img_shape=img_shape, dryrun=args.dryrun)
    else:
        local_gradient_steps = args.accumulation
        local_lr = 1e-4
        input_parameters = inversefed.reconstruction_algorithms.loss_steps(
            model, ground_truth, labels, lr=local_lr, local_steps=local_gradient_steps
        )
        input_parameters = [p.detach() for p in input_parameters]

        # Run reconstruction in different precision?
        if args.dtype != "float":
            if args.dtype in ["double", "float64"]:
                setup["dtype"] = torch.double
            elif args.dtype in ["half", "float16"]:
                setup["dtype"] = torch.half
            else:
                raise ValueError(f"Unknown data type argument {args.dtype}.")
            print(f"Model and input parameter moved to {args.dtype}-precision.")
            ground_truth = ground_truth.to(**setup)
            dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]
            input_parameters = [g.to(**setup) for g in input_parameters]
            model.to(**setup)
            model.eval()

        config = dict(
            signed=args.signed,
            boxed=args.boxed,
            cost_fn=args.cost_fn,
            indices=args.indices,
            weights=args.weights,
            lr=1,
            optim=args.optimizer,
            restarts=args.restarts,
            max_iterations=24000, #24000
            total_variation=args.tv,
            init=args.init,
            filter="none",
            lr_decay=True,
            scoring_choice=args.scoring_choice,
        )

        rec_machine = inversefed.FedAvgReconstructor(
            model, (dm, ds), local_gradient_steps, local_lr, config, num_images=args.num_images, use_updates=True
        )
        output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)

    # Compute stats

    lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
    lpips_score = 99.0
    for i in range(args.num_images):
        score = lpips_loss(output[i], ground_truth[i]).squeeze().item()
        if score < lpips_score:
            lpips_score = score


    def calc_ssim(target, recon):
        '''tructural similarity index'''
        multichannel = True
        tt = transforms.ToPILImage()
        target1 = utils.make_grid(target)
        recon1 = utils.make_grid(recon)
        target_np = np.array(tt(target1))
        recon_np = np.array(tt(recon1))
        # target = target.permute(0, 2, 3, 1).cpu()
        # print(target.shape)c
        # target = target.mul(255).byte()
        # target_np = target.numpy()
        # recon = recon.permute(0, 2, 3, 1).cpu()
        # recon = recon.mul(255).byte()
        # recon_np = recon.numpy()
        ssim_score = ssim(
            target_np, recon_np,
            data_range=255, multichannel=multichannel)

        return ssim_score


    print(output.shape)
    print(ground_truth.shape)

    output_denormalized1 = torch.clamp(output * ds + dm, 0, 1)
    ground_truth_denormalized1 = torch.clamp(ground_truth * ds + dm, 0, 1)
    #ground_truth1 = (ground_truth-0.5).true_divide(0.5)
    #ssim_score = calc_ssim(output, ground_truth)
    ssim_score = calc_ssim(output_denormalized1, ground_truth_denormalized1)
    test_mse = (output - ground_truth).pow(2).mean().item()
    feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)
    # Save the resulting image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)

        if args.num_images == 1:
            if args.dataset == "myimagenet":
                rec_filename = f"{args.model}80_trained_imagenet_newb1_df_{args.cost_fn}-{args.target_id}.png"
            else:
                rec_filename = (
                    f'{validloader.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
                    f"{args.model}_{args.cost_fn}-{args.target_id}.png"
                )
        else:
            rec_filename = (
                #f"{args.model}trained_224imagenet_b8_tv_BN_noGroup_nodf{args.cost_fn}-{args.target_id}.png"
                f"dlg_224imagenet{args.cost_fn}-{args.target_id}.png"
            )
        torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)

        if args.num_images == 1:
            if args.dataset == "myimagenet":
                gt_filename = f"1_ground_truth-{args.target_id}.png"
            else:
                gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png"
        else:
            gt_filename = f"224myimagenet_b8_ground_truth-{args.target_id}.png"

        #gt_filename = f"{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png"
        torchvision.utils.save_image(gt_denormalized, os.path.join(args.image_path, gt_filename))
    else:
        rec_filename = None
        gt_filename = None

    # Save to a table:
    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} | lpips_score: {lpips_score:2.4e} | ssim_score: {ssim_score:2.4e}")

    inversefed.utils.save_to_table(
        args.table_path,
        name=f"exp_{args.name}",
        dryrun=args.dryrun,
        model=args.model,
        dataset=args.dataset,
        trained=args.trained_model,
        accumulation=args.accumulation,
        restarts=args.restarts,
        OPTIM=args.optim,
        cost_fn=args.cost_fn,
        indices=args.indices,
        weights=args.weights,
        scoring=args.scoring_choice,
        init=args.init,
        tv=args.tv,
        rec_loss=stats["opt"],
        psnr=test_psnr,
        test_mse=test_mse,
        feat_mse=feat_mse,
        target_id=target_id,
        seed=model_seed,
        timing=str(datetime.timedelta(seconds=time.time() - start_time)),
        dtype=setup["dtype"],
        epochs=defs.epochs,
        val_acc=None,
        rec_img=rec_filename,
        gt_img=gt_filename,
    )

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print("---------------------------------------------------")
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print("-------------Job finished.-------------------------")

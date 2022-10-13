# -*- coding: utf-8 -*-
import torch

import models.model_sstsc.utils.transforms as transforms
from models.model_sstsc.dataloader.ucr2018 import UCR2018, MultiUCR2018
from models.model_sstsc.model.model_RelationalReasoning import RelationalReasoning_SupInter
from models.model_sstsc.model.model_backbone import SimConv4
from torch.utils.data.sampler import SubsetRandomSampler


def train_SemiIntra(x_train, y_train, x_val, y_val, x_test, y_test,opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        temp_class=2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        temp_class=3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        temp_class=4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        temp_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        temp_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        temp_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        temp_class = 8

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = RelationalReasoning_SupIntra(backbone, feature_size, opt.nb_class, temp_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)

    train_set = MultiUCR2018_Intra(data=x_train, targets=y_train, K=K,
                               transform=train_transform, transform_cut=cut_piece,
                               totensor_transform=tensor_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = model.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                     train_loader_label=train_loader_label,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return test_acc, acc_unlabel, best_epoch


def train_SemiInter(x_train, y_train, x_val, y_val, x_test, y_test,opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = RelationalReasoning_SupInter(backbone, feature_size, opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform)

    train_set = MultiUCR2018(data=x_train, targets=y_train, K=K, transform=train_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = model.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                     train_loader_label=train_loader_label,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return test_acc, acc_unlabel, best_epoch

def train_SemiInter_OL(x_train, y_train, x_val, y_val, x_test, y_test,opt, labeled_index_in_Train):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = RelationalReasoning_SupInter(backbone, feature_size, opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform)

    train_set = MultiUCR2018(data=x_train, targets=y_train, K=K, transform=train_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    # train_dataset_size = len(train_set_labeled)
    # partial_size = int(opt.label_ratio * train_dataset_size)
    # train_ids = list(range(train_dataset_size))
    # np.random.shuffle(train_ids)
    # train_sampler = SubsetRandomSampler(train_ids[:partial_size])
    train_sampler = SubsetRandomSampler(labeled_index_in_Train)

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    # torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch, best_predictions, best_val_prediction, best_val_true_labels = model.train_oline(tot_epochs=tot_epochs, train_loader=train_loader,
                                     train_loader_label=train_loader_label,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    # torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    return test_acc, acc_unlabel, best_epoch, best_predictions, best_val_prediction, best_val_true_labels

def train_Forecasting(x_train, y_train, x_val, y_val, x_test, y_test,opt):
    # Hyper-parameters of the simulation  仿真的超参数
    K = opt.K  # tot augmentations, in the paper K=32 for CIFAR10/100
    batch_size = opt.batch_size  # 64 has been used in the paper
    tot_epochs = opt.epochs  # 200 has been used in the paper
    feature_size = opt.feature_size  # number of units for the Conv4 backbone
    ckpt_dir = opt.ckpt_dir

    # Those are the transformations used in the paper
    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)  # CIFAR10
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    slide_win = transforms.SlideWindow(stride=opt.stride, horizon=opt.horizon)

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()  # simple CNN with 64 linear output units
    model = Forecasting(backbone, feature_size, int(opt.horizon*x_train.shape[1]), opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)

    train_set = MultiUCR2018_Forecast(data=x_train, targets=y_train, K=K,
                               transform=train_transform, transform_cut=slide_win,
                               totensor_transform=tensor_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = model.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                     train_loader_label=train_loader_label,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    # return acc_max, epoch_max
    return test_acc, acc_unlabel, best_epoch




def train_SemiInterPF(x_train, y_train, x_val, y_val, x_test, y_test,opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir

    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    cutPF = transforms.CutPF(sigma=opt.alpha)
    cutPF_transform = transforms.Compose([cutPF])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = RelationalReasoning_SupPF(backbone, feature_size, opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform_label)

    train_set = MultiUCR2018_PF(data=x_train, targets=y_train, K=K,
                                transform=train_transform,
                                transform_cuts=cutPF_transform,
                                totensor_transform=tensor_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, acc_unlabel, best_epoch = model.train(tot_epochs=tot_epochs, train_loader=train_loader,
                                     train_loader_label=train_loader_label,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    # return acc_max, epoch_max
    return test_acc, acc_unlabel, best_epoch


def train_pseudo(x_train, y_train, x_val, y_val, x_test, y_test,opt):
    K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    num_workers=opt.num_workers
    prob = 0.2  # Transform Probability
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    cutPF = transforms.CutPF(sigma=opt.alpha)
    cutPF_transform = transforms.Compose([cutPF])
    w_transform= transforms.Scaling(sigma=0.1, p=1.0)

    W_transform= transforms.Compose([w_transform])
    # cutout = transforms.Cutout(sigma=0.1, p=1.0)
    # jitter = transforms.Jitter(sigma=0.1, p=1.0)
    # scaling = transforms.Scaling(sigma=0.1, p=1.0)
    # magnitude_warp = transforms.MagnitudeWrap(sigma=0.1, knot=4, p=1.0)
    # time_warp = transforms.TimeWarp(sigma=0.5, knot=8, p=1.0)
    # window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=1.0)
    # window_warp = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    s_transform = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    S_transform = transforms.Compose([s_transform])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = pseudo(backbone, feature_size, opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train,
                                        transform=train_transform_label,
                                )


    train_set=UCR2018(data=x_train, targets=y_train,
                                transform=train_transform_label,
                                 )
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])


    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, best_val,acc_epoch_ws, best_epoch = model.train(tot_epochs=tot_epochs,
                                     train_loader_label=train_loader_label,
                                     train_ws_loader=train_loader,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    # return acc_max, epoch_max
    return test_acc, best_val,acc_epoch_ws, best_epoch

def train_vat(l_x_train, l_y_train, x_val, y_val, x_test, y_test,opt):
    prob = opt.K  # Transform Probability K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    num_workers = opt.num_workers
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])

    # cutPF = transforms.CutPF(sigma=opt.alpha)
    # cutPF_transform = transforms.Compose([cutPF])
    w_transform= transforms.Scaling(sigma=0.1, p=1.0)

    W_transform= transforms.Compose([w_transform])
    # cutout = transforms.Cutout(sigma=0.1, p=1.0)
    # jitter = transforms.Jitter(sigma=0.1, p=1.0)
    # scaling = transforms.Scaling(sigma=0.1, p=1.0)
    # magnitude_warp = transforms.MagnitudeWrap(sigma=0.1, knot=4, p=1.0)
    # time_warp = transforms.TimeWarp(sigma=0.5, knot=8, p=1.0)
    # window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=1.0)
    # window_warp = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    s_transform = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    S_transform = transforms.Compose([s_transform])

    tensor_transform = transforms.ToTensor()


    backbone = SimConv4().cuda()
    model = vat(backbone, feature_size, opt.n_class).cuda()

    train_set_labeled = UCR2018(data=l_x_train, targets=l_y_train,
                                        transform=train_transform_label
                                )


    train_set_ws=UCR2018(data=l_x_train, targets=l_x_train,
                                transform=train_transform_label

                                 )
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)


    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                     batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=True,
                                                     shuffle=True
                                                    )

    train_ws_loader = torch.utils.data.DataLoader(train_set_ws,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True)
    val_loader  = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, best_val,acc_epoch_ws, best_epoch = model.train(tot_epochs=tot_epochs,
                                     train_loader_label=train_loader_label,
                                     train_ws_loader=train_ws_loader,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    # return acc_max, epoch_max
    return test_acc, best_val,acc_epoch_ws, best_epoch

def train_pi(x_train, y_train,x_val, y_val, x_test, y_test,opt):

    prob = opt.K  # Transform Probability K = opt.K
    batch_size = opt.batch_size  # 128 has been used in the paper
    tot_epochs = opt.epochs  # 400 has been used in the paper
    feature_size = opt.feature_size
    ckpt_dir = opt.ckpt_dir
    num_workers=opt.num_workers
    raw = transforms.Raw()
    cutout = transforms.Cutout(sigma=0.1, p=prob)
    jitter = transforms.Jitter(sigma=0.2, p=prob)
    scaling = transforms.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': [raw]}



    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms.Compose(transforms_targets)
    train_transform_label = transforms.Compose(transforms_targets + [transforms.ToTensor()])


    cutPF = transforms.CutPF(sigma=opt.alpha)
    cutPF_transform = transforms.Compose([cutPF])
    w_transform= transforms.Scaling(sigma=0.1, p=1.0)

    W_transform= transforms.Compose([w_transform])
    # cutout = transforms.Cutout(sigma=0.1, p=1.0)
    # jitter = transforms.Jitter(sigma=0.1, p=1.0)
    # scaling = transforms.Scaling(sigma=0.1, p=1.0)
    # magnitude_warp = transforms.MagnitudeWrap(sigma=0.1, knot=4, p=1.0)
    # time_warp = transforms.TimeWarp(sigma=0.5, knot=8, p=1.0)
    # window_slice = transforms.WindowSlice(reduce_ratio=0.8, p=1.0)
    # window_warp = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    s_transform = transforms.WindowWarp(window_ratio=0.5, scales=(0.5, 2), p=1.0)
    S_transform = transforms.Compose([s_transform])

    tensor_transform = transforms.ToTensor()

    backbone = SimConv4().cuda()
    model = pi(backbone, feature_size, opt.nb_class).cuda()

    train_set_labeled = UCR2018(data=x_train, targets=y_train,
                                        transform=train_transform_label)


    train_set_ws=UCR20181_for_pi(data=x_train, targets=y_train,
                                transform=train_transform_label
                                 )
    val_set = UCR2018(data=x_val, targets=y_val, transform=tensor_transform)
    test_set = UCR2018(data=x_test, targets=y_test, transform=tensor_transform)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                     batch_size=batch_size,
                                                     sampler=train_sampler)
    train_loader = torch.utils.data.DataLoader(train_set_ws,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    torch.save(model.backbone.state_dict(), '{}/backbone_init.tar'.format(ckpt_dir))
    test_acc, best_val,acc_epoch_ws, best_epoch = model.train(tot_epochs=tot_epochs,
                                     train_loader_label=train_loader_label,
                                     train_ws_loader=train_loader,
                                     val_loader=val_loader,
                                     test_loader=test_loader,
                                     opt=opt)

    torch.save(model.backbone.state_dict(), '{}/backbone_last.tar'.format(ckpt_dir))

    # return acc_max, epoch_max
    return test_acc, best_val,acc_epoch_ws, best_epoch





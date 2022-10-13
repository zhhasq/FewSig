# -*- coding: utf-8 -*-

import torch
import models.model_sstsc.utils.transforms as transforms
from models.model_sstsc.dataloader.ucr2018 import UCR2018
import torch.utils.data as data
from models.model_sstsc.optim.pytorchtools import EarlyStopping
from models.model_sstsc.model.model_backbone import SimConv4


def evaluation(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, ckpt, opt, ckpt_tosave=None):
    # no augmentations used for linear evaluation
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = UCR2018(data=x_train, targets=y_train, transform=transform_lineval)
    val_set_lineval = UCR2018(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = UCR2018(data=x_test, targets=y_test, transform=transform_lineval)

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=128, shuffle=True)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False)
    signal_length = x_train.shape[1]

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model
    # backbone_lineval = OS_CNN(signal_length).cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, nb_class).cuda()
    # linear_layer = torch.nn.Linear(backbone_lineval.rep_dim, nb_class).cuda()

    checkpoint = torch.load(ckpt, map_location='cpu')
    backbone_lineval.load_state_dict(checkpoint)
    if ckpt_tosave:
        torch.save(backbone_lineval.state_dict(), ckpt_tosave)

    optimizer = torch.optim.Adam(linear_layer.parameters(), lr=opt.learning_rate_test)
    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience_test, verbose=True)
    best_acc = 0
    best_epoch = 0

    print('Linear evaluation')
    for epoch in range(opt.epochs_test):
        linear_layer.train()
        backbone_lineval.eval()

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data).detach()
            output = linear_layer(output)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_trains.append(accuracy.item())

        print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, opt.model_name, loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, None)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return test_acc, best_epoch





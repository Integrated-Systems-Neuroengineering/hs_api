import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import argparse
from torch.utils.data import DataLoader
from spikingjelly.activation_based import encoding, functional
from spikingjelly.datasets import padded_sequence_mask
import time
import os
import datetime
import matplotlib.pyplot as plt
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.activation_based.neuron import IFNode, LIFNode
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import profiler
import numpy as np
from tqdm import tqdm


def isSNNLayer(layer):
    return (
        isinstance(layer, MultiStepLIFNode)
        or isinstance(layer, LIFNode)
        or isinstance(layer, IFNode)
    )


def train(args, net, train_loader, test_loader, device, scaler):
    """Given a net and train_loader, this helper function trains the network for the given epochs
        It can also resume from checkpoint

    Args:
        args: command line arguments
        net: the network to be trained
        train_loader: pytorch train DataLoader object
        test_loader: pytorch test DataLoader object
        device: cpu or cuda
        scaler: used for amp mixed percision training

    """
    start_epoch = 0
    max_test_acc = -1

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    encoder, writer = None, None
    if args.encoder:
        encoder = encoding.PoissonEncoder()
        # encoder = encoding.LatencyEncoder(args.T)

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        max_test_acc = checkpoint["max_test_acc"]

    if args.writer:
        writer = SummaryWriter(args.out_dir)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, args.targets).float()
            out_fr = 0.0
            if args.encoder:
                if args.amp:
                    with amp.autocast():
                        if args.transformer:
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)
                        if args.dvs:
                            # [N, T, C, H, W] -> [T, N, C, H, W]
                            img = img.transpose(0, 1)
                            for t in range(args.T):
                                encoded_img = encoder(img[t])
                                out_fr += net(encoded_img)
                        else:
                            for t in range(args.T):
                                encoded_img = encoder(img)
                                out_fr += net(encoded_img)
                else:
                    if args.transformer:
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    if args.dvs:
                        # [N, T, C, H, W] -> [T, N, C, H, W]
                        img = img.transpose(0, 1)
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            out_fr += net(encoded_img)
                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)

            else:
                if args.transformer:
                    out_fr += net(img)
                if args.dvs:
                    # [N, T, C, H, W] -> [T, N, C, H, W]
                    img = img.transpose(0, 1)
                    out_fr += net(img)
                else:
                    for t in range(args.T):
                        out_fr += net(img)

            out_fr = out_fr / args.T
            loss = loss_fun(out_fr, label_onehot)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        if args.writer:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.0

                if args.encoder:
                    if args.transformer:
                        encoded_img = encoder(img)
                        out_fr += net(encoded_img)
                    if args.dvs:
                        img = img.transpose(0, 1)
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            out_fr += net(encoded_img)
                    else:
                        for t in range(args.T):
                            encoded_img = encoder(img)
                            out_fr += net(encoded_img)
                else:
                    if args.dvs:
                        img = img.transpose(0, 1)
                        for t in range(args.T):
                            out_fr += net(img[t])
                    else:
                        for t in range(args.T):
                            out_fr += net(img)

                out_fr = out_fr / args.T

                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            if args.writer:
                writer.add_scalar("test_loss", test_loss, epoch)
                writer.add_scalar("test_acc", test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_max_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )
            if args.transformer:
                checkpoint_ssa = {"ssa": net.block[0].attn.state_dict()}
                torch.save(
                    checkpoint_ssa,
                    os.path.join(
                        args.out_dir,
                        f"checkpoint_max_ssa_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth",
                    ),
                )

        torch.save(
            checkpoint,
            os.path.join(
                args.out_dir,
                f"checkpoint_latest_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth",
            ),
        )

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )


def train_DVS(args, net, train_loader, test_loader, device, scaler):
    """Similar function to train but used for DVS dataset only to speed up the inference"""
    start_epoch = 0
    max_test_acc = -1

    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    encoder = encoding.PoissonEncoder()

    # using two writers to overlay the plot
    writer = SummaryWriter("log_dvs")

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        max_test_acc = checkpoint["max_test_acc"]

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            img = img.transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, args.targets).float()
            out_fr = 0.0

            with amp.autocast():
                for t in range(args.T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / args.T
                loss = loss_fun(out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                img = img.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, args.targets).float()
                out_fr = 0.0

                for t in range(args.T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / args.T
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples

            writer.add_scalars(
                "loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch
            )
            writer.add_scalars(
                "acc", {"train_acc": train_acc, "test_acc": test_acc}, epoch
            )

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_max_T_{args.T}_C_{args.channels}_lr_{args.lr}_opt_{args.opt}.pth",
                ),
            )

        torch.save(
            checkpoint,
            os.path.join(
                args.out_dir,
                f"checkpoint_latest_T_{args.T}_C_{args.channels}_lr_{args.lr}_opt_{args.opt}.pth",
            ),
        )

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )


def train_DVS_Mul(args, net, train_loader, test_loader, device, scaler):
    """Similar function to train_DVS but no encoder and use multistep mode from spikingjelly"""
    start_epoch = 0
    max_test_acc = -1

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="./log_ibm")

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        max_test_acc = checkpoint["max_test_acc"]

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            img = img.transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, args.targets).float()
            out_fr = 0.0

            with amp.autocast():
                out_fr = net(img).mean(0)
                loss = loss_fun(out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                img = img.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, args.targets).float()

                out_fr = net(img).mean(0)
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples

            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_acc", test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_max_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )

        torch.save(
            checkpoint,
            os.path.join(
                args.out_dir,
                f"checkpoint_latest_T_{args.T}_C_{args.channels}_lr_{args.lr}.pth",
            ),
        )

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )


def train_DVS_Time(args, net, train_loader, test_loader, device, scaler, save_every=0):
    """Similar function to train_DVS but using a DVS dataset that has been splitted into frames
    using fix time duration.
    """
    start_epoch = 0
    max_test_acc = -1

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    encoder = encoding.PoissonEncoder()

    # using two writers to overlay the plot
    writer = SummaryWriter("log_dvs_time")

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        max_test_acc = checkpoint["max_test_acc"]

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label, _ in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            img = img.transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, args.targets).float()
            T = img.shape[0]
            out_fr = 0.0

            with amp.autocast():
                for t in range(T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / T
                loss = loss_fun(out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label, _ in test_loader:
                img = img.to(device)
                img = img.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, args.targets).float()
                out_fr = 0.0
                T = img.shape[0]

                for t in range(T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / T
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples

            writer.add_scalars(
                "loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch
            )
            writer.add_scalars(
                "acc", {"train_acc": train_acc, "test_acc": test_acc}, epoch
            )

        # Print min, max, mean, median, std of last synaptic layer's weights (like quantization methods)
        last_layer = None
        for module in reversed(list(net.modules())):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                last_layer = module
                break
        if last_layer is not None:
            w = last_layer.weight.detach().cpu().numpy().flatten()
            print(f"[LAST LAYER WEIGHTS] min: {w.min():.6g}, max: {w.max():.6g}, mean: {w.mean():.6g}, median: {np.median(w):.6g}, std: {w.std():.6g}")

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }


        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_max_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )

        torch.save(
            checkpoint,
            os.path.join(
                args.out_dir,
                f"checkpoint_latest_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
            ),
        )

        # Save every N epochs if requested
        if save_every and save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_epoch_{epoch+1}_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )


def train_DVS_Time_with_plot(args, net, train_loader, test_loader, device, scaler, save_every=0):
    """Identical to train_DVS_Time but also creates and saves training plots"""
    import matplotlib.pyplot as plt
    
    start_epoch = 0
    max_test_acc = -1
    
    # Lists to store training history for plotting
    train_acc_history = []
    test_acc_history = []
    train_loss_history = []
    test_loss_history = []
    epochs_list = []

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    encoder = encoding.PoissonEncoder()

    # using two writers to overlay the plot
    writer = SummaryWriter("log_dvs_time")

    if args.resume_path != "":
        checkpoint = torch.load(args.resume_path, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]
        max_test_acc = checkpoint["max_test_acc"]

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label, _ in train_loader:
            optimizer.zero_grad()
            img = img.to(device)
            img = img.transpose(0, 1)
            label = label.to(device)
            label_onehot = F.one_hot(label, args.targets).float()
            T = img.shape[0]
            out_fr = 0.0

            with amp.autocast():
                for t in range(T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / T
                loss = loss_fun(out_fr, label_onehot)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label, _ in test_loader:
                img = img.to(device)
                img = img.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, args.targets).float()
                out_fr = 0.0
                T = img.shape[0]

                for t in range(T):
                    encoded_img = encoder(img[t])
                    out_fr += net(encoded_img)

                out_fr = out_fr / T
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples

            writer.add_scalars(
                "loss", {"train_loss": train_loss, "test_loss": test_loss}, epoch
            )
            writer.add_scalars(
                "acc", {"train_acc": train_acc, "test_acc": test_acc}, epoch
            )

        # Print min, max, mean, median, std of last layer's weights after each epoch
        last_weight = None
        for p in reversed(list(net.parameters())):
            if p.requires_grad and p.data.ndim > 0:
                last_weight = p.data.detach().cpu().numpy().flatten()
                break
        if last_weight is not None:
            print(f"[LAST LAYER WEIGHTS] min: {last_weight.min():.6g}, max: {last_weight.max():.6g}, mean: {last_weight.mean():.6g}, median: {np.median(last_weight):.6g}, std: {last_weight.std():.6g}")

        # Store training history for plotting
        epochs_list.append(epoch)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "max_test_acc": max_test_acc,
        }


        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_max_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )

        torch.save(
            checkpoint,
            os.path.join(
                args.out_dir,
                f"checkpoint_latest_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
            ),
        )

        # Save every N epochs if requested
        if save_every and save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(
                checkpoint,
                os.path.join(
                    args.out_dir,
                    f"checkpoint_epoch_{epoch+1}_T_{T}_C_{args.channels}_lr_{args.lr}.pth",
                ),
            )

        print(
            f"epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}"
        )
        print(
            f"train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s"
        )
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n'
        )

    # Create and save training plots
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Accuracy over time
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_acc_history, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs_list, test_acc_history, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss over time
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_loss_history, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs_list, test_loss_history, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot to output directory
    plot_path = os.path.join(args.out_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    # Also save as PDF for better quality
    pdf_path = os.path.join(args.out_dir, 'training_curves.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Training curves saved to: {pdf_path}")
    
    plt.close()  # Close the figure to free memory


def test_DVS_Time(args, net, test_loader, device, scaler):
    """Similar function to train_DVS but using a DVS dataset that has been splitted into frames
    using fix time duration.
    """
    start_epoch = 0
    max_test_acc = -1

    loss_fun = nn.CrossEntropyLoss()

    encoder = encoding.PoissonEncoder()

    # using two writers to overlay the plot
    writer = SummaryWriter("log_dvs_time")

    for epoch in range(start_epoch, args.epochs):
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0

        with torch.no_grad():
            for img, label, _ in test_loader:
                img = img.to(device)
                img = img.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, args.targets).float()
                out_fr = 0.0
                T = img.shape[0]

                for t in range(T):
                    encoded_img = encoder(img[t])
                    netout = net(encoded_img)
                    out_fr += netout

                out_fr = out_fr / T
                loss = loss_fun(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

            test_time = time.time()
            # test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples

    print("accuracy: " + str(test_acc))

    return test_acc, test_loss


def validate(args, net, test_loader, device, converter=None):
    """Given a net and test_loader, this helper function test the network for on the sepecified
        platform. If testing a HiAER Spike compatible network on Python Simulation or FPGA, a converter
        object is passed in to call the helper function.

    Args:
        args: command line arguments
        net: the network to be trained
        test_loader: pytorch train DataLoader object
        device: cpu or cuda
        converter: converter object to test a HiAER Spike network on software simulation/FPGA

    """
    start_time = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    writer, encoder = None, None
    if args.writer:
        writer = SummaryWriter(args.out_dir)
    encoder = None
    if args.encoder:
        encoder = encoding.PoissonEncoder()

    loss_fun = nn.MSELoss()
    # loss_fun = nn.CrossEntropyLoss()

    if args.cri:
        # dvs: [B, T, C, H, W] regualr img: [B, C, H, W]
        for img, label in test_loader:
            label_onehot = F.one_hot(label, 10).float()
            out_fr = 0.0

            cri_input = None

            if args.dvs:
                if args.encoder:
                    encoded_img = encoder(img)
                    cri_input = converter.input_converter(encoded_img)
                else:
                    cri_input = converter.input_converter(img)
            else:
                if args.encoder:
                    img_repeats = img.repeat(args.T, 1, 1, 1, 1)
                    cri_input = []
                    for t in range(args.T):
                        encoded_img = encoder(img_repeats[t])
                        cri_input.append(encoded_img)
                    cri_input = converter.input_converter(
                        torch.stack(cri_input).transpose(0, 1)
                    )
                else:
                    cri_input = converter.input_converter(
                        img.repeat(args.T, 1, 1, 1, 1)
                    )

            if args.hardware:
                out_fr = torch.tensor(
                    converter.run_CRI_hw(cri_input, net), dtype=float
                ).to(device)
            else:
                out_fr = torch.tensor(
                    converter.run_CRI_sw(cri_input, net), dtype=float
                ).to(device)

            loss = loss_fun(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()

            test_acc += (out_fr.argmax(1) == label).float().sum().item()

        test_time = time.time()
        test_speed = test_samples / (test_time - start_time)
        test_loss /= test_samples
        test_acc /= test_samples

        if args.writer:
            writer.add_scalar("test_loss", test_loss)
            writer.add_scalar("test_acc", test_acc)

    else:

        net.eval()

        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.0

                if args.dvs:
                    img = img.transpose(0, 1)
                    if args.encoder:
                        for t in range(args.T):
                            encoded_img = encoder(img[t])
                            netout = net(encoded_img)
                            out_fr += netout
                            print(netout)
                    else:
                        for t in range(args.T):
                            out_fr += net(img[t])
                else:
                    if args.encoder:
                        encoded_img = encoder(img)
                        out_fr += net(img)
                    else:
                        out_fr += net(img)
                # breakpoint()
                out_fr = out_fr / args.T

                loss = loss_fun(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()

                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)  # reset the membrane potential after each img
            test_time = time.time()
            test_speed = test_samples / (test_time - start_time)
            test_loss /= test_samples
            test_acc /= test_samples

            if args.writer:
                writer.add_scalar("test_loss", test_loss)
                writer.add_scalar("test_acc", test_acc)

    print(f"test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")


def sw_comp_DVS(args, net, test_loader, device, torchnet, converter=None, one_batch_only=False):
    """Similar function to validate but used for DVS dataset only"""

    start_time = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    writer = SummaryWriter(log_dir="./log_hardware")
    encoder = encoding.PoissonEncoder()

    loss_fun = nn.MSELoss()
    torchnet.eval()
    for batch_idx, (img, label, x_len) in enumerate(tqdm(test_loader)):
        # T appears to be different for different batches
        img = img.transpose(0, 1)  # [B, T, C, H, W] -> [T, B, C, H, W]
        label_onehot = F.one_hot(label, args.targets).float()
        out_fr = 0.0

        cri_input = []

        for t in img:
            encoded_img = encoder(t)
            cri_input.append(encoded_img)
            netout = torchnet(encoded_img)
            print("size" + str(len(img)))
            print(netout)

        torch_input = cri_input

        # breakpoint()
        cri_input = torch.stack(cri_input)
        # breakpoint()
        cri_input = cri_input.transpose(0, 1)
        # looks like the converter wants batch in the first dimension
        cri_input = converter.input_converter(cri_input)
        out_fr = torch.tensor(converter.run_CRI_sw(cri_input, net), dtype=float).to(device)
        # Debug: print raw output, argmax, and one-hot for each sample
        print("[DEBUG] Raw output (before one-hot):")
        print(out_fr)
        for idx, elem in enumerate(out_fr):
            hot = torch.argmax(elem)
            print(f"[DEBUG] Sample {idx}: argmax={hot.item()}, raw={elem.tolist()}")
            row = torch.zeros_like(elem)
            row[hot] = 1
            print(f"[DEBUG] Sample {idx}: one-hot={row.tolist()}")
            out_fr[idx] = row
        print("[DEBUG] out_fr (after one-hot): " + str(out_fr))

        # breakpoint()

        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()

        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        print("acc: " + str(test_acc / test_samples))

        if one_batch_only:
            break
    # breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    breakpoint()
    writer.add_scalar("test_loss", test_loss)
    writer.add_scalar("test_acc", test_acc)

    print(f"test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")


def validate_DVS(args, net, test_loader, device, converter=None):
    """Similar function to validate but used for DVS dataset only"""

    start_time = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    writer = SummaryWriter(log_dir="./log_hardware")
    encoder = encoding.PoissonEncoder()

    loss_fun = nn.MSELoss()

    for img, label, x_len in tqdm(test_loader):
        # T appears to be different for different batches
        img = img.transpose(0, 1)  # [B, T, C, H, W] -> [T, B, C, H, W]
        label_onehot = F.one_hot(label, args.targets).float()
        out_fr = 0.0

        cri_input = []

        for t in img:
            encoded_img = encoder(t)
            cri_input.append(encoded_img)

        cri_input = torch.stack(cri_input)
        # breakpoint()
        cri_input = cri_input.transpose(0, 1)
        # looks like the converter wants batch in the first dimension
        cri_input = converter.input_converter(cri_input)
        out_fr = torch.tensor(converter.run_CRI_sw(cri_input, net), dtype=float).to(
            device
        )
        # breakpoint()
        for idx, elem in enumerate(out_fr):
            row = torch.zeros_like(elem)
            hot = torch.argmax(elem)
            row[hot] = 1
            out_fr[idx] = row

        # breakpoint()

        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()

        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        print("acc: " + str(test_acc / test_samples))
        breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples

    writer.add_scalar("test_loss", test_loss)
    writer.add_scalar("test_acc", test_acc)

    print(f"test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")


def validate_DVS_HW(args, net, test_loader, device, converter=None):
    """Similar function to validate but used for DVS dataset only"""

    start_time = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    writer = SummaryWriter(log_dir="./log_hardware")
    encoder = encoding.PoissonEncoder()

    loss_fun = nn.MSELoss()

    for img, label, x_len in tqdm(test_loader):
        # T appears to be different for different batches
        img = img.transpose(0, 1)  # [B, T, C, H, W] -> [T, B, C, H, W]
        label_onehot = F.one_hot(label, args.targets).float()
        out_fr = 0.0

        cri_input = []

        for t in img:
            print("did one image")
            encoded_img = encoder(t)
            cri_input.append(encoded_img)

        cri_input = torch.stack(cri_input)
        # breakpoint()
        cri_input = cri_input.transpose(0, 1)
        # looks like the converter wants batch in the first dimension
        cri_input = converter.input_converter(cri_input)
        out_fr = torch.tensor(converter.run_CRI_hw(cri_input, net), dtype=float).to(
            device
        )
        # breakpoint()
        for idx, elem in enumerate(out_fr):
            row = torch.zeros_like(elem)
            hot = torch.argmax(elem)
            row[hot] = 1
            out_fr[idx] = row

        # breakpoint()

        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()

        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        print("acc: " + str(test_acc / test_samples))
        breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples

    writer.add_scalar("test_loss", test_loss)
    writer.add_scalar("test_acc", test_acc)

    print(f"test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")

#krish

def infer_cri_params(model, synaptic_types=(nn.Conv2d, nn.Linear, nn.AvgPool2d)):
    """
    Infers the input layer index, number of SNN layers, and output layer index
    by scanning only the top-level modules (no recursion).
    """
    # Always look inside the first nn.Sequential block if present
    layers = None
    for mod in model.children():
        if isinstance(mod, nn.Sequential):
            layers = list(mod.children())
            break
    if layers is None:
        layers = list(model.children())

    input_layer = None
    output_layer = None
    snn_layers = 0
    for idx, layer in enumerate(layers):
        if isinstance(layer, synaptic_types):
            if input_layer is None:
                input_layer = idx
            output_layer = idx
            snn_layers += 1
    return input_layer, snn_layers, output_layer


def infer_cri_params_submodules(model, synaptic_types=(nn.Conv2d, nn.Linear)):
    """
    Infers the number of SNN layers and the output layer index for CRI_Converter, but krish custom ones(like converter_krish_flattened)

    Parameters
    ----------
    model: nn.Module
        The PyTorch model, expected to have a .conv_fc Sequential block.

    synaptic_types: tuple
        A tuple of layer types to be treated as synaptic layers,
        e.g., (nn.Conv2d, nn.Linear). These are the layers counted
        toward snn_layers and used to determine output_layer.

    Returns
    -------
    input_layer : int
        Index (within model.conv_fc) of the first synaptic layer.

    snn_layers : int
        Number of synaptic layers (e.g., Conv2d or Linear).

    output_layer : int
        Index (within model.conv_fc) of the last synaptic layer.
    """
    input_layer = -1
    snn_layers = 0
    output_layer = -1
    layer_index = 0
    synaptic_indices = []

    def traverse_layers(module):
        nonlocal input_layer, snn_layers, output_layer, layer_index
        for layer in module.children():
            # If the layer has submodules, recurse into them (like converter)
            if len(list(layer.children())) > 0:
                traverse_layers(layer)
            else:
                if isinstance(layer, synaptic_types):
                    if snn_layers == 0:
                        input_layer = layer_index
                    snn_layers += 1
                    output_layer = layer_index
                    print(f"[infer_cri_params] Synaptic layer found at idx={layer_index}: {type(layer).__name__}")
                layer_index += 1

    if hasattr(model, 'conv_fc'):
        traverse_layers(model.conv_fc)
    else:
        traverse_layers(model)

    if output_layer != -1:
        print(f"[infer_cri_params] Output layer index: {output_layer}, type: (see above for type)")
    else:
        print("[infer_cri_params] No synaptic output layer found.")

    return input_layer, snn_layers, output_layer

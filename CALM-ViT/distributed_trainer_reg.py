import os
from time import time, sleep
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import pyspark
import CALM_ViT_V2 as rvh
import gc
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.torch.distributor import TorchDistributor
from torchvision.datasets import ImageNet
from subprocess import call
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import InterpolationMode
import sys

parent_dir = "/config"

def train(initializer, optimizer, scheduler,
          use_gpu=True, dataset=None, epochs=15, batch_size=128):
    import torch
    import os
    import gc
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    import torch.nn as nn
    import time
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, default_collate
    from torch.amp import autocast, GradScaler

    print("Training on device", os.environ['RANK'])

    gc.collect()
    if use_gpu: torch.cuda.empty_cache()
    os.environ['NCCL_DEBUG'] = 'ERROR'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    dist.init_process_group(backend='nccl' if use_gpu else 'gloo')

    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    model = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True, seed=2006)
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    criterion_huber = nn.HuberLoss(delta=1.0)
    scaler = GradScaler(device="cuda",enabled=use_gpu)
    # attention_maps = None
    # batch = None
    # y_pred = None
    # y_actual = None
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # dataset.reshuffle()
        # dataset.train = True
        model.train()
        for i, (x, y) in enumerate(dataloader):
            # batch = x
            x = x.to(device)
            y = y.to(device)
            # Compute classification and reconstruction loss on the batch
            with autocast(device_type="cuda", enabled=use_gpu):
                y_hat, kl_loss = model(x)
                img = y_hat.reshape(-1, 224, 224, 3)
                img = img.permute(0, 3, 1, 2)
                # loss_1 = criterion(y_hat.squeeze(), y) # The labels need to be floating point
                loss_2 = criterion_huber(img, x)
                # img_flat = img.reshape(-1, 224 * 224 * 3)
                # x_flat = x.reshape(-1, 224 * 224 * 3)
                # img_log = torch.nn.functional.log_softmax(img_flat, dim=1)
                # x_soft = torch.nn.functional.softmax(x_flat, dim=1)
                # loss_3 = criterion_kl(img_log, x_soft)
                # loss = loss_1 # + loss_2 + loss_3
                loss = loss_2 + kl_loss * 0.1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25, error_if_nonfinite=False)
            # _, predicted = torch.max(y_hat.data, 1)
            # correct = (predicted == y).sum().item()
            # accuracy = correct / y.size(0)
            # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, Accuracy: {accuracy}")
            scaler.step(optimizer)
            scaler.update()
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Device: [{global_rank}, {local_rank}], Loss: {loss}")
        if global_rank == 0 and local_rank == 0:
            torch.save(model.module.state_dict(), f"{parent_dir}/Codebase/models/model_reg.pth")
            rvh.save_samples(img)
            print("Model saved to models/model_reg.pth")
            # save_samples(batch, attention_maps, y_pred, y_actual)
        if scheduler is not None:
            scheduler.step()
            if global_rank == 0 and local_rank == 0:
                print(f"Epoch {epoch + 1} scheduler step!")
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module

if __name__ == "__main__":
    start = time()
    call(['mkdir', '-p', f'{parent_dir}/Codebase/models'])
    call(['mkdir', '-p', f'{parent_dir}/Codebase/samples'])
    spark = SparkSession.builder.appName("CALM_ViT_Training").getOrCreate()
    rvh.save_samples(torch.zeros((1, 3, 224, 224)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ColorJitter(brightness=(0.5, 1), contrast=(0.5, 1), saturation=(0.5, 1), hue=(-0.125, 0.125)),
        transforms.RandomSolarize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageNet(
        root="/dataset/imagenet/",
        split="train",
        transform=transform
    )
    lr = 0
    for i in range(10):
        lr += 3.1e-5
        distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
        model = rvh.ViT(device, type=8, heads=12, seq_length=224, in_features=672,
                 dim_step=48, mean_var_hidden=224,
                 seq_len_step=16, seq_len_reduce=128, out_features=672,
                 force_reduce=False, generate=True)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(f"{parent_dir}/Codebase/models/model_reg.pth", map_location=device, weights_only=True))
            print("Loaded existing model weights from model_reg.pth")
        except:
            print("No existing model weights found, starting fresh training.")
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        model = distributor.run(
            train,
            model,
            optimizer=opt,
            scheduler=None,
            use_gpu=True,
            dataset=dataset,
            epochs=5,
            batch_size=544
        )
        torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_reg.pth")
        torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_reg_{i}.pth")
        print(f"Model saved to {parent_dir}/Codebase/models/model_reg.pth")
        print(f"Model saved to {parent_dir}/Codebase/models/model_reg_{i}.pth")
    for i in range(10):
        lr += 3.1e-4
        distributor = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True)
        model = rvh.ViT(device, type=8, heads=12, seq_length=224, in_features=672,
                 dim_step=48, mean_var_hidden=224,
                 seq_len_step=16, seq_len_reduce=128, out_features=672,
                 force_reduce=False, generate=True)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model.load_state_dict(torch.load(f"{parent_dir}/Codebase/models/model_reg.pth", map_location=device, weights_only=True))
            print("Loaded existing model weights from model_reg.pth")
        except:
            print("No existing model weights found, starting fresh training.")
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        model = distributor.run(
            train,
            model,
            optimizer=opt,
            scheduler=None,
            use_gpu=True,
            dataset=dataset,
            epochs=5,
            batch_size=544
        )
        torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_reg.pth")
        torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_reg_{i + 1}.pth")
        print(f"Model saved to {parent_dir}/Codebase/models/model_reg.pth")
        print(f"Model saved to {parent_dir}/Codebase/models/model_reg_{i + 1}.pth")
    print(f"Model saved to {parent_dir}/Codebase/models/model_reg_fnl.pth")
    print(f"Time taken: {time() - start}")
    sleep(30)
    spark.stop()
import os
from time import time, sleep
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import pyspark
import reverse_ViT_hybrid as rvh
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
    from torch.utils.data import DataLoader, DistributedSampler

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
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    model = initializer
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    sampler = DistributedSampler(dataset, shuffle=True)
    sampler.set_epoch(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    criterion = torch.nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
    # attention_maps = None
    # batch = None
    # y_pred = None
    # y_actual = None
    model.train()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        # dataset.reshuffle()
        # dataset.train = True
        epoch_loss = 0.0
        predicted = 0
        model.train()
        for i, (x, y) in enumerate(dataloader):
            # batch = x
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # Compute classification and reconstruction loss on the batch
            y_hat, img = model(x)
            loss = criterion(y_hat.squeeze(), y)
            loss += criterion_mse(img, x)
            img_flat = img.reshape(-1, 256 * 256 * 3)
            x_flat = x.reshape(-1, 256 * 256 * 3)
            loss += criterion_kl(torch.log_softmax(img_flat, dim=1), torch.softmax(x_flat, dim=1))
            loss.backward()
            # with torch.no_grad():
            #     # attention_maps = model.module.get_attention_maps(x)
            #     img_detached = img.detach()
            # # Compute adversarial loss on the batch
            # fake_y_hat, img = model(img_detached)
            # fake_y = torch.ones_like(y).float()
            # sa_loss = criterion_binary(fake_y_hat.squeeze(), fake_y)
            # sa_loss += criterion_mse(img, img_detached)
            # img_flat = img.reshape(-1, 256 * 256 * 3)
            # img_detached_flat = img_detached.reshape(-1, 256 * 256 * 3)
            # sa_loss += criterion_kl(torch.log_softmax(img_flat, dim=1), torch.softmax(img_detached_flat, dim=1))
            # sa_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(y_hat.data, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Device: [{global_rank}, {local_rank}], Loss: {loss}, Accuracy: {accuracy * 100:.4f}%")
            if global_rank == 0 and local_rank == 0 and i % 250 == 0:
                torch.save(model.module.state_dict(), f"{parent_dir}/Codebase/models/model_b.pth")
                rvh.save_samples(img)
                print("Model saved to models/model_b.pth")
                # save_samples(batch, attention_maps, y_pred, y_actual)
        if scheduler is not None:
            scheduler.step()
    model = model.to("cpu")
    dist.destroy_process_group()
    return model.module

if __name__ == "__main__":
    start = time()
    call(['mkdir', '-p', f'{parent_dir}/Codebase/models'])
    call(['mkdir', '-p', f'{parent_dir}/Codebase/samples'])
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()
    distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)
    model = rvh.initialize_vit(torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights=f"")
    print(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.v2.Compose([
        torchvision.transforms.v2.Resize((256, 256)),
        torchvision.transforms.v2.RandomHorizontalFlip(),
        torchvision.transforms.v2.ToImage(),
        torchvision.transforms.v2.ToDtype(dtype=torch.float32, scale=True),
        torchvision.transforms.v2.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.v2.Lambda(lambda x: torch.nn.functional.tanh(x))
    ])
    opt = optim.Adam(model.parameters(), lr=0.0001)
    dataset = ImageNet(
        root=parent_dir + "/imagenet/",
        split="train",
        transform=transform
    )
    model = distributor.run(
        train,
        model,
        optimizer=opt,
        scheduler=None,
        use_gpu=True,
        dataset=dataset,
        epochs=3,
        batch_size=48
    )
    torch.save(model.state_dict(), f"{parent_dir}/Codebase/models/model_b.pth")
    print(f"Model saved to {parent_dir}/Codebase/models/model_b.pth")
    # sleep(30)
    # save_samples(batch, attention_maps, y_pred, y_actual)
    print(f"Time taken: {time() - start}")
    sc.stop()
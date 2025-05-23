import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from models import PAMNet, PAMNet_a, PAMNet_s, Config
from utils import EMA
from datasets import QMOF  # This is your QMOF dataset implementation.

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, loader, ema, device):
    mae = 0
    ema.assign(model)
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        mae += (output - data.y).abs().sum().item()
    # ema.resume(model)
    return mae / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=480, help='Random seed.')
    parser.add_argument('--model', type=str, default='PAMNet', choices=['PAMNet', 'PAMNet_a', 'PAMNet_s'], help='Model to be used')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=3, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Dimension for embeddings.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    args = parser.parse_args()

    # Initialize Weights & Biases
    wandb.init(project='QMOF_training', config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)

    # Load and filter dataset: only graphs with <=80 nodes
    full_dataset = QMOF(root='data/qmof_database', cutoff=5.0, target_column="bandgap")
    idxs = [i for i, d in enumerate(full_dataset) if d.num_nodes <= 80]
    dataset = full_dataset[idxs]
    print(f"Number of graphs in dataset: {len(dataset)}")
    # Split into training, validation, and test sets.
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_dataset = dataset[:n_train]
    val_dataset   = dataset[n_train:n_train + n_val]
    test_dataset  = dataset[n_train + n_val:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    config = Config(dataset="QMOF", dim=args.dim, n_layer=args.n_layer, cutoff_l=5.0, cutoff_g=5.0)
    if args.model == 'PAMNet':
        model = PAMNet(config).to(device)
    elif args.model == 'PAMNet_a':
        model = PAMNet_a(config).to(device)
    else:
        model = PAMNet_s(config).to(device)

    print("Number of model parameters:", count_parameters(model))
    wandb.watch(model, log='all')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)
    ema = EMA(model, decay=0.999)

    best_val_loss = float('inf')
    train_losses, val_losses, test_losses = [], [], []

    with tqdm(total=args.epochs, desc="Epochs", unit="epoch") as epoch_bar:
        for epoch in range(args.epochs):
            model.train()
            loss_all = 0.0
            count = 0
            step = 0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", leave=False) as batch_bar:
                for data in batch_bar:
                    data = data.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.l1_loss(output, data.y)
                    loss_all += loss.item() * data.num_graphs
                    count += data.num_graphs
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1000)
                    optimizer.step()

                    curr_epoch = epoch + float(step) / (len(train_dataset) / args.batch_size)
                    scheduler_warmup.step(curr_epoch)

                    step += 1
                    batch_bar.set_postfix(loss=f"{loss.item():.6f}")

            loss_epoch = loss_all / count
            val_loss = test(model, val_loader, ema, device)
            test_loss = test(model, test_loader, ema, device)

            train_losses.append(loss_epoch)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

            # Log to wandb
            wandb.log({
                'epoch': epoch + 1,
                'train_mae': loss_epoch,
                'val_mae': val_loss,
                'test_mae': test_loss,
                'lr': optimizer.param_groups[0]['lr']
            })

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_folder = "./save/QMOF"
                os.makedirs(save_folder, exist_ok=True)
                torch.save(model.state_dict(), osp.join(save_folder, "best_model.pt"))

            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_folder = "./save/QMOF/checkpoints"
                os.makedirs(ckpt_folder, exist_ok=True)
                torch.save(model.state_dict(), osp.join(ckpt_folder, f"epoch_{epoch+1}.pt"))

            tqdm.write(f"Epoch: {epoch+1:03d}, Train MAE: {loss_epoch:.7f}, Val MAE: {val_loss:.7f}, Test MAE: {test_loss:.7f}")
            epoch_bar.update(1)
            epoch_bar.set_postfix({"Train MAE": f"{loss_epoch:.7f}", "Val MAE": f"{val_loss:.7f}"})

    print("Best Validation MAE:", best_val_loss)
    print("Test MAE:", test_loss)

    # Plot and save
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), train_losses, label="Train MAE")
    plt.plot(range(1, args.epochs+1), val_losses, label="Validation MAE")
    plt.plot(range(1, args.epochs+1), test_losses, label="Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training, Validation, and Test MAE per Epoch")
    plt.legend()
    os.makedirs("save/QMOF", exist_ok=True)
    plt.savefig("save/QMOF/train_mae.jpg")

if __name__ == "__main__":
    main()

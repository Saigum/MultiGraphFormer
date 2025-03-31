import os
import os.path as osp
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
import matplotlib.pyplot as plt

from models import PAMNet, PAMNet_s, Config, PAMNet_a
from utils import EMA
from datasets import QM9

# Import Ignite components for engine, metrics and early stopping
from ignite.engine import Engine, Events
from ignite.metrics import Average  # Use Average instead of Mean
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar  # Import the progress bar from ignite.contrib

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU number.')
    parser.add_argument('--seed', type=int, default=480, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='QM9', help='Dataset to be used')
    parser.add_argument('--model', type=str, default='PAMNet', choices=['PAMNet','PAMNet_a', 'PAMNet_s'], help='Model to be used')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss).')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of hidden layers.')
    parser.add_argument('--dim', type=int, default=128, help='Size of input hidden units.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--target', type=int, default=7, help='Index of target for prediction')
    parser.add_argument('--cutoff_l', type=float, default=5.0, help='cutoff in local layer')
    parser.add_argument('--cutoff_g', type=float, default=5.0, help='cutoff in global layer')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    class MyTransform(object):
        def __call__(self, data):
            target = args.target
            # Adjust target index if needed
            if target in [7, 8, 9, 10]:
                target = target + 5
            data.y = data.y[:, target]
            return data

    # Create dataset
    path = osp.join('.', 'data', args.dataset)
    dataset = QM9(path, transform=MyTransform()).shuffle()

    # Split dataset
    train_dataset = dataset[:110000]
    val_dataset = dataset[110000:120000]
    test_dataset = dataset[120000:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Data loaded!")

    config = Config(dataset=args.dataset, dim=args.dim, n_layer=args.n_layer, cutoff_l=args.cutoff_l, cutoff_g=args.cutoff_g)

    if args.model == 'PAMNet':
        model = PAMNet(config).to(device)
    elif args.model == "PAMNet_a":
        model = PAMNet_a(config).to(device)
    else:
        model = PAMNet_s(config).to(device)
    print("Number of model parameters: ", count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

    ema = EMA(model, decay=0.999)

    # --- Ignite Engines ---
    # Training step for each batch
    def train_step(engine, batch):
        model.train()
        data = batch.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.l1_loss(output, data.y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
        optimizer.step()
        ema(model)
        return loss.item()

    trainer = Engine(train_step)

    # Attach progress bar to trainer
    ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x})

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_scheduler(engine):
        # Calculate fractional epoch: (iteration - 1)/#iters_per_epoch
        current_iter = engine.state.iteration
        iters_per_epoch = len(train_loader)
        frac_epoch = (current_iter - 1) / iters_per_epoch
        scheduler_warmup.step(frac_epoch)

    # Validation step (used for both validation and testing)
    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            data = batch.to(device)
            output = model(data)
            loss = F.l1_loss(output, data.y)
        return loss.item()

    evaluator = Engine(validation_step)
    Average(output_transform=lambda x: x).attach(evaluator, "val_loss")

    # Define early stopping based on the validation loss.
    def score_function(engine):
        val_loss = engine.state.metrics["val_loss"]
        return -val_loss  # negative because EarlyStopping expects higher scores to be better

    early_stopping_handler = EarlyStopping(patience=args.patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

    train_losses = []
    val_losses = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics["val_loss"]
        train_losses.append(engine.state.output)  # Using the last batch loss of the epoch
        val_losses.append(val_loss)
        print(f"Epoch: {engine.state.epoch:03d}, Val MAE: {val_loss:.7f}")

    print("Start training!")
    trainer.run(train_loader, max_epochs=args.epochs)

    evaluator.run(test_loader)
    test_loss = evaluator.state.metrics["val_loss"]
    print('Testing MAE:', test_loss)

    # Plot training and validation losses per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_losses) + 1), train_losses, label="Train Loss (last batch)")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / MAE")
    plt.title("Training and Validation MAE per Epoch")
    plt.legend()
    save_folder = osp.join(".", "save", args.dataset)
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(osp.join(save_folder, "train_val_mae.jpg"))

if __name__ == "__main__":
    main()

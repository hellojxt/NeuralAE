import sys

sys.path.append("./")
from src.dataset import AudioDataset
from src.model import ConditionalTCN
from src.loss import AudioLoss
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/musdb18")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sample_rate", type=int, default=24000)
parser.add_argument("--nblocks", type=int, default=4)
parser.add_argument("--kernel_size", type=int, default=13)
parser.add_argument("--dilation_growth", type=int, default=8)
parser.add_argument("--channel_width", type=int, default=64)
parser.add_argument("--noncausal", action="store_true")
parser.add_argument("--max_epochs", type=int, default=100)

args = parser.parse_args()


train_dataset = AudioDataset(args.data_dir + "/train", sample_rate=args.sample_rate)
test_dataset = AudioDataset(args.data_dir + "/test", sample_rate=args.sample_rate)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)

model = ConditionalTCN(
    args.sample_rate,
    nblocks=args.nblocks,
    kernel_size=args.kernel_size,
    dialation_growth=args.dilation_growth,
    channel_width=args.channel_width,
    causal=not args.noncausal,
)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_fn = AudioLoss(alpha=100)


def step(train=True):
    if train:
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = test_loader

    total_loss = 0
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / len(loader)


def train():
    return step(train=True)


@torch.no_grad()
def test():
    return step(train=False)


best_loss = float("inf")

for epoch in range(args.max_epochs):
    train_loss = train()
    test_loss = test()
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "model.pt")

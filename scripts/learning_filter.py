import sys

sys.path.append("./")
from src.dataset import AudioDataset
from src.model import ConditionalTCN
from src.loss import AudioLoss
import torch
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchaudio

writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset/musdb18/reverb")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--sample_rate", type=int, default=24000)
parser.add_argument("--nblocks", type=int, default=5)
parser.add_argument("--kernel_size", type=int, default=13)
parser.add_argument("--dilation_growth", type=int, default=8)
parser.add_argument("--channel_width", type=int, default=8)
parser.add_argument("--noncausal", action="store_true")
parser.add_argument("--max_epochs", type=int, default=100)

args = parser.parse_args()

train_dataset = AudioDataset(args.data_dir + "/train_patch")
test_dataset = AudioDataset(args.data_dir + "/test_patch")
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
    dilation_growth=args.dilation_growth,
    channel_width=args.channel_width,
    causal=not args.noncausal,
)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_fn = AudioLoss(alpha=100)


log_sample_num = 4


def audio_to_sepctrogram(x):
    x = x.reshape(-1)
    tfm1 = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=1024, hop_length=256
    ).cuda()
    tfm2 = torchaudio.transforms.AmplitudeToDB().cuda()
    x = tfm1(x)
    x = tfm2(x)
    return x


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spec(x, y, y_hat):
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    axs[0].imshow(x.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axs[1].imshow(y.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axs[2].imshow(
        y_hat.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma"
    )
    axs[0].set_title("Input")
    axs[1].set_title("Target")
    axs[2].set_title("Prediction")
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    fig.tight_layout(pad=0)
    return fig


def log_sample(x, y, y_hat, log_sample_idx):
    writer.add_audio(
        f"x_{log_sample_idx}",
        x[0],
        epoch,
        sample_rate=args.sample_rate,
    )
    writer.add_audio(
        f"y_{log_sample_idx}",
        y[0],
        epoch,
        sample_rate=args.sample_rate,
    )
    writer.add_audio(
        f"y_hat_{log_sample_idx}",
        y_hat[0],
        epoch,
        sample_rate=args.sample_rate,
    )
    writer.add_figure(
        f"spec_{log_sample_idx}",
        plot_spec(
            audio_to_sepctrogram(x[0]),
            audio_to_sepctrogram(y[0]),
            audio_to_sepctrogram(y_hat[0]),
        ),
        epoch,
    )


def step(train=True):
    if train:
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = test_loader

    total_loss = 0
    log_sample_idx = 0
    for x, y in tqdm(loader):
        x = x.cuda()
        y = y.cuda()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (
            log_sample_idx % 16 == 0
            and log_sample_idx / 16 < log_sample_num
            and not train
        ):
            log_sample(x, y, y_hat, log_sample_idx // 16)
        log_sample_idx += 1

    return total_loss / len(loader)


best_loss = float("inf")

for epoch in range(args.max_epochs):
    train_loss = step(train=True)
    with torch.no_grad():
        test_loss = step(train=False)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "model.pt")

writer.close()

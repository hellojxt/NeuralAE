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
import os
import torchsummary
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="m4singer")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--sample_rate", type=int, default=44100)
parser.add_argument("--nblocks", type=int, default=6)
parser.add_argument("--kernel_size", type=int, default=13)
parser.add_argument("--dilation_growth", type=int, default=6)
parser.add_argument("--channel_width", type=int, default=64)
parser.add_argument("--noncausal", action="store_false")
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--pretrained", type=str, default=None)
parser.add_argument("--tag", type=str, default="test")

args = parser.parse_args()
data_dir = f"dataset/{args.dataset}"
log_dir = "output/" + args.tag
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

train_dataset = AudioDataset(data_dir, "train")
test_dataset = AudioDataset(data_dir, "test")
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
    condition=False,
)
model = model.cuda()

with open(log_dir + '/torchsummary.txt', 'w') as f:
    report = torchsummary.summary(model, (1, 5*args.sample_rate))
    f.write(report.__repr__())
    f.write(f"\nreceptive: {(model.receptive_field/model.sample_rate)*1e3:0.3f} ms")
# torchsummary.summary(model, (1, 5*args.sample_rate))

if args.pretrained is not None:
    model.load_state_dict(torch.load(args.pretrained))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.4)
loss_fn = AudioLoss(alpha=100)


log_sample_num = 8


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
    fig, axs = plt.subplots(4, 1, figsize=(15, 8))
    axs[0].imshow(x.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axs[1].imshow(y.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    axs[2].imshow(
        y_hat.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma"
    )
    axs[3].imshow(
        y_hat.detach().cpu().numpy() - y.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="magma"
    )
    SNR = (10 * torch.log10(torch.sum(y**2) / torch.sum((y - y_hat) ** 2))).item()
    SNR_input = (10 * torch.log10(torch.sum(y**2) / torch.sum((x - y) ** 2))).item()
    axs[0].set_title("Input")
    axs[1].set_title("Target")
    axs[2].set_title(f"Predicted (SNR: {SNR:.2f} dB, SNR_input: {SNR_input:.2f} dB)")
    axs[3].set_title("Residual")
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    axs[2].set_xticks([])
    axs[3].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[3].set_yticks([])
    fig.tight_layout(pad=0)
    return fig

def log_sample(x, y, y_hat, log_sample_idx):
    # save audio
    torchaudio.save(
        f"{log_dir}/{log_sample_idx}_x.wav",
        x[0].cpu(),
        sample_rate=args.sample_rate,
    )
    torchaudio.save(
        f"{log_dir}/{log_sample_idx}_y.wav",
        y[0].cpu(),
        sample_rate=args.sample_rate,
    )
    torchaudio.save(
        f"{log_dir}/{log_sample_idx}_y_hat.wav",
        y_hat[0].cpu(),
        sample_rate=args.sample_rate,
    )
    fig = plot_spec(
        audio_to_sepctrogram(x[0]),
        audio_to_sepctrogram(y[0]),
        audio_to_sepctrogram(y_hat[0]),
    )
    # save figure
    fig.savefig(f"{log_dir}/{log_sample_idx}.png")
    plt.close(fig)


def remove_head(x):
    return x[:, :, int(x.shape[2]*0.1):]

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
        y = y[:, :, -y_hat.shape[2]:]
        x = remove_head(x)
        y = remove_head(y)
        y_hat = remove_head(y_hat)
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
test_loss_history = []

for epoch in range(args.max_epochs):
    train_loss = step(train=True)
    test_start_time = time()
    with torch.no_grad():
        test_loss = step(train=False)
    test_end_time = time()
    inference_time = test_end_time - test_start_time
    inference_speed = len(test_dataset) * 5 / inference_time
    writer.add_scalar("Inference Speed", inference_speed, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    # print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    scheduler.step(test_loss)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), f"{log_dir}/model.pt")
    test_loss_history.append(test_loss)

fig = plt.figure()
plt.plot(test_loss_history)
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.savefig(f"{log_dir}/loss.png")
writer.close()

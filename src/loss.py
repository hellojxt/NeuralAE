import auraloss
import torch.nn as nn


class AudioLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.freq_loss = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048, 8192, 32768],
            hop_sizes=[16, 64, 256, 1024, 4096, 16384],
            win_lengths=[32, 128, 512, 2048, 8192, 32768],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )
        self.time_domain_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, output, target):
        return self.freq_loss(output, target) + self.alpha * self.time_domain_loss(
            output, target
        )

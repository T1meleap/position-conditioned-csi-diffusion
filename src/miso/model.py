import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(timesteps, dim):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return F.pad(result, (0, dim % 2))


class ResBlock1D(nn.Module):
    def __init__(self, incoming, outgoing, embedding_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, incoming), incoming)
        self.conv1 = nn.Conv1d(incoming, outgoing, 3, padding=1)
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, outgoing))
        self.norm2 = nn.GroupNorm(min(8, outgoing), outgoing)
        self.conv2 = nn.Conv1d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, value, embedding):
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.embedding(embedding)[:, :, None]
        return self.conv2(F.silu(self.norm2(hidden))) + self.skip(value)


class CSIUNet(nn.Module):
    def __init__(self, num_tx=8, base_channels=64, embedding_dim=128):
        super().__init__()
        self.num_tx, self.embedding_dim = num_tx, embedding_dim
        self.time_mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * 2), nn.SiLU(), nn.Linear(embedding_dim * 2, embedding_dim))
        self.coordinate_mlp = nn.Sequential(nn.Linear(3, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
        self.input = nn.Conv1d(4 * num_tx + 1, base_channels, 3, padding=1)
        self.down_block_1 = ResBlock1D(base_channels, base_channels, embedding_dim)
        self.downsample_1 = nn.Conv1d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.down_block_2 = ResBlock1D(base_channels * 2, base_channels * 2, embedding_dim)
        self.downsample_2 = nn.Conv1d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.middle_1 = ResBlock1D(base_channels * 4, base_channels * 4, embedding_dim)
        self.middle_2 = ResBlock1D(base_channels * 4, base_channels * 4, embedding_dim)
        self.upsample_2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up_block_2 = ResBlock1D(base_channels * 4, base_channels * 2, embedding_dim)
        self.upsample_1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up_block_1 = ResBlock1D(base_channels * 2, base_channels, embedding_dim)
        self.output = nn.Sequential(nn.GroupNorm(min(8, base_channels), base_channels), nn.SiLU(), nn.Conv1d(base_channels, 2 * num_tx, 3, padding=1))

    def forward(self, noisy, timesteps, condition, coordinates):
        if noisy.shape[1] != 2 * self.num_tx or condition.shape[1] != 2 * self.num_tx + 1:
            raise ValueError("Antenna/channel dimensions disagree")
        embedding = self.time_mlp(timestep_embedding(timesteps, self.embedding_dim)) + self.coordinate_mlp(coordinates)
        hidden = self.input(torch.cat([noisy, condition], dim=1))
        skip1 = self.down_block_1(hidden, embedding)
        skip2 = self.down_block_2(self.downsample_1(skip1), embedding)
        hidden = self.middle_2(self.middle_1(self.downsample_2(skip2), embedding), embedding)
        hidden = self.up_block_2(torch.cat([self.upsample_2(hidden), skip2], dim=1), embedding)
        hidden = self.up_block_1(torch.cat([self.upsample_1(hidden), skip1], dim=1), embedding)
        return self.output(hidden)


class IndependentCSIUNet(nn.Module):
    """Shared single-antenna network; no cross-antenna feature mixing."""
    def __init__(self, num_tx=8, base_channels=64):
        super().__init__()
        self.num_tx = num_tx
        self.single = CSIUNet(1, base_channels)

    def forward(self, noisy, timesteps, condition, coordinates):
        b, _, k = noisy.shape
        n = self.num_tx
        x = torch.stack([noisy[:, :n], noisy[:, n:]], dim=2).reshape(b * n, 2, k)
        observed = torch.stack([condition[:, :n], condition[:, n:2*n]], dim=2).reshape(b * n, 2, k)
        mask = condition[:, -1:].repeat_interleave(n, dim=0)
        y = self.single(x, timesteps.repeat_interleave(n), torch.cat([observed, mask], dim=1), coordinates.repeat_interleave(n, dim=0))
        y = y.reshape(b, n, 2, k)
        return torch.cat([y[:, :, 0], y[:, :, 1]], dim=1)


class DDPM(nn.Module):
    def __init__(self, network, timesteps=200):
        super().__init__()
        self.model, self.timesteps = network, timesteps
        beta = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float64)
        roots = torch.cumprod(1 - beta, dim=0).sqrt()
        roots = (roots - roots[-1]) * roots[0] / (roots[0] - roots[-1])
        bars = roots.square()
        beta = (1 - bars / F.pad(bars[:-1], (1, 0), value=1)).float()
        alpha = 1 - beta
        bars = torch.cumprod(alpha, dim=0)
        previous = F.pad(bars[:-1], (1, 0), value=1)
        self.register_buffer("root", bars.sqrt())
        self.register_buffer("sigma", (1 - bars).sqrt())
        self.register_buffer("variance", (beta * (1 - previous) / (1 - bars)).clamp_min(1e-20))
        self.register_buffer("clean_coefficient", beta * previous.sqrt() / (1 - bars))
        self.register_buffer("noisy_coefficient", (1 - previous) * alpha.sqrt() / (1 - bars))

    @staticmethod
    def at(value, t):
        return value[t][:, None, None]

    def loss(self, clean, condition, coordinates, t=None, noise=None):
        if t is None:
            t = torch.randint(self.timesteps, (len(clean),), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        root, sigma = self.at(self.root, t), self.at(self.sigma, t)
        noisy = root * clean + sigma * noise
        target = root * noise - sigma * clean
        prediction = self.model(noisy, t, condition, coordinates).float()
        unknown = 1 - condition[:, -1:]
        return ((prediction - target).square() * unknown).sum() / (unknown.sum() * clean.shape[1]).clamp_min(1)

    @torch.no_grad()
    def sample(self, condition, coordinates, seed):
        g = torch.Generator(device=condition.device).manual_seed(seed)
        observed, mask = condition[:, :-1], condition[:, -1:]
        value = torch.randn(observed.shape, generator=g, device=condition.device)
        known_noise = torch.randn(observed.shape, generator=g, device=condition.device)
        value = value * (1 - mask) + known_noise * mask
        for step in reversed(range(self.timesteps)):
            t = torch.full((len(value),), step, device=value.device, dtype=torch.long)
            velocity = self.model(value, t, condition, coordinates).float()
            clean = self.root[step] * value - self.sigma[step] * velocity
            value = self.clean_coefficient[step] * clean + self.noisy_coefficient[step] * value
            if step:
                value = value + self.variance[step].sqrt() * torch.randn(value.shape, device=value.device, generator=g)
                known = self.root[step - 1] * observed + self.sigma[step - 1] * known_noise
            else:
                known = observed
            value = value * (1 - mask) + known * mask
        return value

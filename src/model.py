import argparse
import importlib.util
import io
import math
import os

import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SyntheticDataset
from environment import set_seed, set_env
from model_paths import build_model_filename

# Note: onnxsim is not imported here (call _require_onnxsim() where it is needed).
# On import, onnxsim runs `pip install onnxruntime` automatically if onnxruntime is
# missing; many parallel jobs doing this at once on compute nodes break the shared environment.


# ===== Training settings =========================================
# Training hyperparameters for the synthetic data (syn).
# Shared by train() (trains models and saves them to files) and
# train_synthetic_model_ephemeral() (throwaway models for the interval-counting experiment).
SYN_BATCH_SIZE = 32
SYN_EPOCHS = 1000
SYN_LR = 3e-4
SYN_NUM_SAMPLES = 1000
SYN_TEST_NUM_TIMESTEPS = 460
SYN_SAMPLING_STEP = 115
SYN_IN_CH = 1

# Offset of the training-data seed for the throwaway models.
# The evaluation data use seeds 0..29 (+2023 for the reference images), so the
# training data are moved to the 10000 range to keep them disjoint from the evaluation data.
TRAIN_DATA_SEED_OFFSET = 10000
# ================================================================


def _require_onnxsim():
    """Check that onnxsim can be used safely and return its simplify function.

    On import, onnxsim tries to run `pip install onnxruntime` if onnxruntime is
    missing. Parallel jobs doing this at the same time corrupt the shared
    environment (site-packages), so we first check only that onnxruntime exists
    and, if it does not, fail immediately with a clear error instead of triggering pip.
    """
    if importlib.util.find_spec("onnxruntime") is None:
        raise RuntimeError(
            "onnxruntime is not available in this Python environment. onnxruntime "
            "is a dependency in pyproject.toml, so check that the job uses the "
            "uv-managed environment (the .venv created by `uv sync` at the project root). "
            "(We stop here because onnxsim's automatic pip install conflicts between "
            "parallel jobs and breaks the environment.)"
        )
    from onnxsim import simplify

    return simplify


def optimize_onnx(input_path, output_path):
    simplify = _require_onnxsim()
    model = onnx.load(input_path)

    model_optimized, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_optimized, output_path)

    print(f"Optimized model saved to {output_path}")


def _pos_encoding(time_idx, output_dim, device):
    t = time_idx
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device).float()
    div_term = torch.exp(i / D * math.log(10000))

    angle = t / div_term
    v = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))

    return v


def pos_encoding(timesteps, output_dim, device):
    batch_size = timesteps.shape[0]
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, act_fn):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act_fn
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch), act_fn, nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, act_fn="relu"):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        if act_fn == "relu":
            act_fn = nn.ReLU()
        elif act_fn == "silu":
            act_fn = nn.SiLU()
        elif act_fn == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        self.down1 = ConvBlock(in_ch, 32, time_embed_dim, act_fn)
        self.down2 = ConvBlock(32, 64, time_embed_dim, act_fn=act_fn)
        self.down3 = ConvBlock(64, 128, time_embed_dim, act_fn=act_fn)
        self.bot1 = ConvBlock(128, 256, time_embed_dim, act_fn=act_fn)
        self.up3 = ConvBlock(128 + 256, 128, time_embed_dim, act_fn=act_fn)
        self.up2 = ConvBlock(64 + 128, 64, time_embed_dim, act_fn=act_fn)
        self.up1 = ConvBlock(32 + 64, 32, time_embed_dim, act_fn=act_fn)
        self.out = nn.Conv2d(32, in_ch, 1)

        self.averagepool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device)

        x1 = self.down1(x, v)
        x = self.averagepool(x1)
        x2 = self.down2(x, v)
        x = self.averagepool(x2)
        x3 = self.down3(x, v)
        x = self.averagepool(x3)
        x = self.bot1(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)

        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        test_num_timesteps,
        sampling_step,
        act_fn="relu",
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        eta=1,
        in_ch=1,
    ):
        super().__init__()
        self.eta = eta
        # compute alpha_bars
        self.num_timesteps = num_timesteps
        self.test_num_timesteps = test_num_timesteps
        self.sampling_step = sampling_step
        self.betas = torch.cat(
            [torch.zeros(1), torch.linspace(beta_start, beta_end, num_timesteps)], dim=0
        )
        self.alpha_bars = (1 - self.betas).cumprod(dim=0).view(-1, 1, 1, 1)
        self.unet = UNet(in_ch=in_ch, act_fn=act_fn)

    def forward(self, input_x):
        x, _ = self.add_noise(input_x, self.test_num_timesteps)
        seq = torch.arange(
            self.test_num_timesteps,
            -1,
            -self.sampling_step,
            device=x.device
        )

        seq_next = torch.full((seq.shape[0],), -1, device=x.device,)
        seq_next[:-1] = seq[1:]
        B = x.size(0)

        for i in range(seq.shape[0]):
            t = torch.full((B,), seq[i], device=x.device, dtype=torch.long)
            next_t = torch.full((B,), seq_next[i], device=x.device, dtype=torch.long)
            at = self.alpha_bars[t + 1].to(x.device)
            at_next = self.alpha_bars[next_t + 1].to(x.device)

            eps_hat = self.unet(x, t).to(x.device)

            x0_t = (x - eps_hat * torch.sqrt(1 - at)) / torch.sqrt(at)
            c1 = self.eta * torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at))
            c2 = torch.sqrt(1 - at_next - c1**2)
            x = (
                torch.sqrt(at_next) * x0_t
                + c1 * torch.randn_like(x).to(x.device)
                + c2 * eps_hat
            )
        return x

    def add_noise(self, x, t):
        eps = torch.randn_like(x).to(x.device)
        at = self.alpha_bars[t + 1].to(x.device)
        at = at.view(-1, 1, 1, 1).to(x.device)
        x_t = torch.sqrt(at) * x + torch.sqrt(1 - at) * eps
        return x_t, eps

    def predict_noise(self, x, t):
        eps_hat = self.unet(x, t).to(x.device)
        return eps_hat


def train_epochs(
    diffusion_model,
    train_loader,
    epochs,
    lr,
    device,
    beta_start=0.0001,
    beta_end=0.02,
    num_timesteps=1000,
):
    """Train for the given number of epochs with the MSE loss of noise prediction (shared training loop).

    Used by both train() (models saved to files) and
    train_synthetic_model_ephemeral() (throwaway models).
    """
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss_sum = 0.0
        cnt = 0
        for x, _, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diffusion_model.num_timesteps, (len(x),)).to(device)
            betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
            at = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1).to(device)
            eps = torch.randn_like(x).to(device)
            x = torch.sqrt(at) * x + torch.sqrt(1 - at) * eps
            eps_hat = diffusion_model.predict_noise(x, t)
            loss = nn.MSELoss()(eps_hat, eps)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1

        loss = loss_sum / cnt
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")


def export_simplified_proto(diffusion_model, img_size: int, in_ch: int) -> onnx.ModelProto:
    """Export the trained model to ONNX, simplify it with onnxsim, and return the ModelProto.

    The export is done in memory (BytesIO), so nothing is written to disk.
    """
    simplify = _require_onnxsim()
    device = torch.device("cpu")
    diffusion_model.eval().to(device)
    dummy_input = torch.randn(1, in_ch, img_size, img_size).to(device)

    buffer = io.BytesIO()
    torch.onnx.export(diffusion_model, dummy_input, buffer)
    model_proto = onnx.load_model_from_string(buffer.getvalue())

    model_simplified, check = simplify(model_proto)
    assert check, "Simplified ONNX model could not be validated"
    return model_simplified


def train_synthetic_model_ephemeral(
    img_size: int,
    model_seed: int,
    act_fn: str = "relu",
) -> onnx.ModelProto:
    """Train one diffusion model on synthetic data for the interval-counting experiment and
    return the simplified ONNX model in memory (ModelProto). **A throwaway model that is never
    written to disk.**

    Following the design "one p-value = one independently trained model", this is called from
    the workers of experiment.py; each worker (one core, one model) trains the model and then
    computes the intervals (the number of threads follows the caller's setting; workers use one
    thread). Weight initialization, shuffling, and noise generation are driven by model_seed, and
    the training-data seed is TRAIN_DATA_SEED_OFFSET + model_seed (disjoint from the evaluation data).
    """
    # Check onnxsim / onnxruntime (needed for the export after training) up front,
    # so that a missing dependency fails immediately rather than after a long training run.
    _require_onnxsim()

    set_seed(model_seed)
    dataset = SyntheticDataset(
        num_samples=SYN_NUM_SAMPLES,
        img_size=img_size,
        signal=0,
        seed=TRAIN_DATA_SEED_OFFSET + model_seed,
    )
    train_loader = DataLoader(dataset, batch_size=SYN_BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion_model = DiffusionModel(
        test_num_timesteps=SYN_TEST_NUM_TIMESTEPS,
        sampling_step=SYN_SAMPLING_STEP,
        act_fn=act_fn,
        in_ch=SYN_IN_CH,
    ).to(device)

    train_epochs(diffusion_model, train_loader, SYN_EPOCHS, SYN_LR, device)
    return export_simplified_proto(diffusion_model, img_size, SYN_IN_CH)


def train(category: str, act_fn: str):
    match category:
        case "syn":
            batch_size = SYN_BATCH_SIZE
            epochs = SYN_EPOCHS
            lr = SYN_LR
            num_samples = SYN_NUM_SAMPLES
            img_sizes = [8, 16, 32, 64]
            test_num_timesteps = SYN_TEST_NUM_TIMESTEPS
            sampling_step = SYN_SAMPLING_STEP
            in_ch = SYN_IN_CH
            save_dir = "../model/syn"

        case _:
            raise ValueError(f"Unsupported category: {category} (only 'syn' is available)")

    os.makedirs(save_dir, exist_ok=True)

    for img_size in img_sizes:
        dataset = SyntheticDataset(
            num_samples=num_samples,
            img_size=img_size,
            signal=0,
            seed=42,
        )

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        assert torch.cuda.is_available(), "CUDA is not available"
        device = torch.device("cuda")
        beta_start = 0.0001
        beta_end = 0.02
        num_timesteps = 1000
        diffusion_model = DiffusionModel(
            test_num_timesteps=test_num_timesteps,
            sampling_step=sampling_step,
            act_fn=act_fn,
            beta_start=beta_start,
            beta_end=beta_end,
            num_timesteps=num_timesteps,
            in_ch=in_ch,
        ).to(device)

        train_epochs(
            diffusion_model,
            train_loader,
            epochs,
            lr,
            device,
            beta_start=beta_start,
            beta_end=beta_end,
            num_timesteps=num_timesteps,
        )

        # export ONNX (file names follow the convention in model_paths; ReLU models carry no activation name)
        device = torch.device("cpu")
        diffusion_model.eval().to(device)
        dummy_input = torch.randn(1, in_ch, img_size, img_size).to(device)
        raw_name = build_model_filename(
            img_size, test_num_timesteps, sampling_step, act_fn, simplified=False
        )
        sim_name = build_model_filename(
            img_size, test_num_timesteps, sampling_step, act_fn, simplified=True
        )
        torch.onnx.export(diffusion_model, dummy_input, f"{save_dir}/{raw_name}")

        # simplify ONNX
        optimize_onnx(f"{save_dir}/{raw_name}", f"{save_dir}/{sim_name}")

if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument("-category", "--category", type=str, default="syn")
    cmdline_parser.add_argument("-act_fn", "--act_fn", type=str, default="relu")

    args, unknowns = cmdline_parser.parse_known_args()
    # Initialize the environment, the number of threads, and the seed only when run from the CLI
    # (not at module level, so that importing this module from experiment.py etc.
    #  does not change the random state or the number of threads).
    set_env()
    torch.set_num_threads(30)
    set_seed(42)
    train(category=args.category, act_fn=args.act_fn)

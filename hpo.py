import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import yaml

MAX_EPOCHS = 2


class CPC(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 slice_length: int = 1024,
                 history_steps: int = 12,
                 drop_rate: float = 0.2,
                 drop_path_rate: float = 0.7):
        super(CPC, self).__init__()

        # define some hyperparameters
        self.slice_length = slice_length
        self.history_steps = history_steps

        # define an encoder
        self.encoder = timm.create_model(
            "resnet18",
            in_chans=in_features,
            num_classes=hidden_features,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # define the autoregressive model
        self.autoregressor = nn.GRU(
            input_size=hidden_features,
            hidden_size=hidden_features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor):
        # x is of shape (B, Ca, C, T)

        # some preprocessing: slice the input into chunks of length slice_length without overlap
        x = x.unfold(3, self.slice_length,
                     self.slice_length).permute(0, 3, 1, 2, 4)  # (B, num_chunks, Ca, C, slice_length)

        # pass the input through the encoder
        B, N, Ca, C, T = x.size()
        x = self.encoder(x.contiguous().view(B*N, Ca, C, T))
        x = x.view(B, N, -1)  # (B, num_chunks, D)

        # pass the history_steps chunks through the autoregressive model
        h0 = torch.zeros(1, x.size(0), x.size(-1)
                         ).to(x.device)  # (num_layers, B, D)

        ct = torch.zeros(B, self.history_steps, x.size(-1)).to(x.device)
        preds = torch.zeros(B, self.history_steps, x.size(-1)).to(x.device)
        for t in range(1, self.history_steps + 1):
            out, h0 = self.autoregressor(x[:, :t], h0)
            ct[:, t-1, :] = out[:, -1, :]
            preds[:, t-1, :] = x[:, t, :]

        return ct, preds, h0


def nt_xent_loss(c_t, z_fut, temperature=1.0):
    # c_t and z_fut is of shape (B, K, D)
    # normalize the vectors
    c_t = F.normalize(c_t, p=2, dim=-1)
    z_fut = F.normalize(z_fut, p=2, dim=-1)

    B, K, _ = z_fut.size()
    # compute the cosine similarity
    logits = torch.einsum("bjd, bkd -> bjk", c_t, z_fut) / \
        temperature  # (B, K, K)
    # positives are the diagonal pairs j == k
    true_labels = torch.arange(K).repeat(B).to(c_t.device)  # (B*K,)

    # compute the loss
    return F.cross_entropy(logits.view(B*K, K), true_labels)


class R22_Dataset(DatasetFolder):
    """
    Ensures every sample tensor ends up the same length (the minimum
    length across your entire dataset), by trimming.
    """

    def __init__(self, root, load_fn, transform=None, **kwargs):
        super().__init__(root, loader=load_fn, transform=transform, **kwargs)
        # 1) scan every file once to find the minimum length
        lengths = []
        for path, _ in self.samples:
            # load only the array header, not the entire payload
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            lengths.append(arr.shape[-1])
        self.min_length = min(lengths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # 2) trim the sample to the minimum length
        sample = sample[..., :self.min_length]
        return sample


def get_dataloader(batch_size=4, num_workers=4):
    def load_fn(path):
        with open(path, "rb") as f:
            data = np.load(f)
            _ = np.load(f, allow_pickle=True).item()
        data = np.stack((data.real, data.imag), axis=1)
        # drop the first 1024 samples
        data = data[:, :, 1024:]
        # drop the last 1024 samples
        data = data[:, :, :-1024]
        # CPC uses a portion of the entire sequence in the loss
        # assuming a autoregressive model consumes 40 steps of 4096 samples
        # then the maximum length of the sequence is 40 * 4096 = 163840
        # randomly sample a portion of the sequence of length 163840
        idx = np.random.randint(0, data.shape[-1] - 163840)
        data = data[:, :,  idx:idx + 163840]
        # # normalize the data
        # data = (data - np.mean(data, axis=(1, 2), keepdims=True)) / \
        #     (np.std(data, axis=(1, 2), keepdims=True) + 1e-8)
        return data.astype(np.float32)

    dataset = R22_Dataset(
        root="./oct10_outdoor_gain_experiments",
        load_fn=load_fn,
        transform=lambda x: torch.from_numpy(x),
        extensions=[".npy"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def objective(trial: optuna.Trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    temperature = trial.suggest_float("temperature", 0.05, 1.0)
    slice_length = trial.suggest_int("slice_length", 256, 4096, step=256)
    history_steps = trial.suggest_int("history_steps", 4, 40, step=4)
    drop_rate = trial.suggest_float("drop_rate", 0.0, 0.5)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.9)

    # create the model
    model = CPC(
        in_features=4,
        hidden_features=128,
        slice_length=slice_length,
        history_steps=history_steps,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    ).to(device)

    # create the optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    dataloader = get_dataloader()

    scaler = GradScaler(device=device.type, enabled=True)

    step = 0
    with tqdm(range(MAX_EPOCHS)) as master_bar:
        for epoch in master_bar:
            model.train()
            avg_loss = 0.0
            with tqdm(dataloader) as pbar:
                for xb in pbar:
                    xb = xb.to(device)
                    optim.zero_grad()

                    with torch.amp.autocast(device_type=device.type,
                                            enabled=True):
                        c_t, z_fut, _ = model(xb)
                        loss = nt_xent_loss(
                            c_t, z_fut, temperature=temperature)

                    scaler.scale(loss).backward()
                    # clip gradients
                    scaler.unscale_(optim)
                    _ = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, norm_type=2
                    )
                    scaler.step(optim)
                    scaler.update()

                    avg_loss += loss.item()

                    pbar.set_postfix({"loss/step": loss.item()})
                    step += 1

            avg_loss /= len(dataloader)
            first_avg_loss = avg_loss if epoch == 0 else first_avg_loss
            master_bar.write(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    trial.report(first_avg_loss - avg_loss, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return first_avg_loss - avg_loss


if __name__ == "__main__":
    # Set up the Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=2
        )
    )

    # Optimize the objective function
    study.optimize(objective, n_trials=10, timeout=60*60*6)

    # Print the best hyperparameters
    print("Best value: ", study.best_value)
    print("Best hyperparameters: ", study.best_params)
    # save best trial to yaml file
    with open("best_trial.yaml", "w") as f:
        yaml.dump(study.best_trial.params, f)

    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(
        study.get_trials(states=[optuna.trial.TrialState.PRUNED])))
    print("Number of complete trials: ", len(
        study.get_trials(states=[optuna.trial.TrialState.COMPLETE])))
    print("Number of failed trials: ", len(
        study.get_trials(states=[optuna.trial.TrialState.FAIL])))

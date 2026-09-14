import sys

import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm


def r2_score(output, target) -> torch.Tensor:
    target_mean = target.mean(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0)
    ss_res = ((output - target) ** 2).sum(dim=0)
    return 1 - (ss_res / ss_tot)


def evaluate(model, data_loader, transform, pbar=False):
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []
    all_targets = []

    if pbar:
        pbar = tqdm(
            total=len(data_loader),
            desc='Evaluation',
            disable=not sys.stderr.isatty(),
        )

    with torch.no_grad():
        for batch in data_loader:
            fingerprints, targets = map(lambda data: data.to(device), batch)
            outputs = model(fingerprints)

            all_outputs.append(outputs)
            all_targets.append(targets)

            if isinstance(pbar, tqdm):
                pbar.update()
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    scaled_targets = transform(all_targets)
    loss = mse_loss(all_outputs, scaled_targets.float(), reduction='none').mean(dim=0)
    loss = loss.sum().item()

    # Undo scaling for metrics, retaining any logarithmic target transform.
    metric_outputs = all_outputs.to(torch.float64) * transform.scale + transform.center
    metric_targets = scaled_targets * transform.scale + transform.center
    r2_scores = r2_score(metric_outputs, metric_targets)
    rmse_scores = torch.sqrt(mse_loss(metric_outputs, metric_targets, reduction='none').mean(dim=0))
    all_outputs = transform(all_outputs.to(torch.float64), inverse=True)

    if isinstance(pbar, tqdm):
        pbar.set_postfix({'loss': loss})
        pbar.close()

    return {
        'loss': loss,
        'r2': r2_scores,
        'rmse': rmse_scores,
        'predictions': all_outputs,
        'targets': all_targets,
    }

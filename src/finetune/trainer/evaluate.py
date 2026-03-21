import torch
from torch.nn.functional import mse_loss, l1_loss
from tqdm import tqdm


def r2_score(output, target) -> torch.Tensor:
    target_mean = target.mean(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0)
    ss_res = ((output - target) ** 2).sum(dim=0)
    return 1 - (ss_res / ss_tot)


def evaluate(model, data_loader, transform, pbar=False, mae=False):
    model.eval()
    device = next(model.parameters()).device
    all_outputs = []
    all_targets = []

    if pbar:
        pbar = tqdm(total=len(data_loader), desc='Validation')

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
    loss = mse_loss(all_outputs, transform(all_targets).float(), reduction='none').mean(dim=0)
    loss = loss.sum().item()

    all_outputs = transform(all_outputs.to(torch.float64), inverse=True)
    r2_scores = r2_score(all_outputs, all_targets)

    if isinstance(pbar, tqdm):
        pbar.set_postfix({'loss': loss})
    
    if mae:
        mae_scores = l1_loss(all_outputs, all_targets, reduction='none').mean(dim=0)
        return loss, r2_scores, mae_scores
    else:
        return loss, r2_scores

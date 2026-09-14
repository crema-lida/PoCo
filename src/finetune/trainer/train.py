import math
import sys

from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .evaluate import evaluate


def train(model, train_loader, transform, optimizer, max_epochs=1, last_epoch=0,
          writer=None, eval_loader=None, patience=None, max_norm=1.0, min_steps=0):
    model.train()
    training_loss = None
    device = next(model.parameters()).device
    target_steps = max(max_epochs * len(train_loader), min_steps)
    total_epochs = math.ceil(target_steps / len(train_loader))
    optimizer_steps = 0
    best_metric = -float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    pbar = tqdm(total=total_epochs, desc='Training' if eval_loader is not None else 'Refitting',
                disable=not sys.stderr.isatty())

    for epoch in range(1, total_epochs + 1):
        model.train()

        for batch in train_loader:
            fingerprints, targets = map(lambda data: data.to(device), batch)

            outputs = model(fingerprints)
            targets = transform(targets).float()

            loss = mse_loss(outputs, targets, reduction='none').mean(dim=0).sum()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer_steps += 1
            optimizer.zero_grad()

            training_loss = loss.item() if training_loss is None else 0.9 * training_loss + 0.1 * loss.item()
            if eval_loader is None and optimizer_steps >= target_steps:
                break
        
        if eval_loader is not None:
            metrics = evaluate(model, eval_loader, transform)
            eval_loss = metrics['loss']
            r2_scores = metrics['r2']
            validation_r2 = float(r2_scores.item())
            if not math.isfinite(validation_r2):
                raise RuntimeError(
                    f'Non-finite validation R2 at epoch {epoch}: {validation_r2}'
                )
            if validation_r2 > best_metric:
                best_metric = validation_r2
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        else:
            eval_loss = None
            r2_scores = None

        pbar.set_postfix_str(f'loss={training_loss:.2e}', refresh=False)
        pbar.update()
        
        if writer is not None:
            global_step = epoch + last_epoch
            losses = {'Training': training_loss}
            if eval_loss is not None:
                losses['Validation'] = eval_loss
            writer.add_scalars('Loss', losses, global_step)
            if r2_scores is not None:
                for i, name in enumerate(train_loader.dataset.properties):
                    writer.add_scalar(f'R2/{name}', r2_scores[i], global_step)

        if (eval_loader is not None and patience is not None
                and optimizer_steps >= min_steps and epochs_no_improve >= patience):
            break

    pbar.close()
    if eval_loader is None:
        return optimizer_steps

    return best_epoch, best_metric

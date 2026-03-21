from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .evaluate import evaluate


def train(model, train_loader, transform, optimizer, scheduler, max_epochs=1,
          last_epoch=0, writer=None, eval_loader=None, max_norm=1.0):
    model.train()
    training_loss = None
    device = next(model.parameters()).device
    pbar = tqdm(total=max_epochs)

    for epoch in range(1, max_epochs + 1):
        pbar.set_description(f'Training')
        model.train()

        for batch in train_loader:
            fingerprints, targets = map(lambda data: data.to(device), batch)

            outputs = model(fingerprints)
            targets = transform(targets).float()

            loss = mse_loss(outputs, targets, reduction='none').mean(dim=0).sum()
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

            training_loss = loss.item() if training_loss is None else 0.9 * training_loss + 0.1 * loss.item()
        
        eval_loss, r2_scores = evaluate(model, eval_loader, transform)
        lr = scheduler.step(r2_scores.cpu().mean().item())

        pbar.set_postfix_str(f'loss={training_loss:.2e}, lr={lr[0]:.2e}', refresh=False)
        pbar.update()
        
        if writer is not None:
            global_step = epoch + last_epoch
            writer.add_scalars('Loss', {'Training': training_loss, 'Validation': eval_loss}, global_step)
            for i, name in enumerate(train_loader.dataset.properties):
                writer.add_scalar(f'R2/{name}', r2_scores[i], global_step)

        if scheduler.early_stop:
            return
    
    scheduler.load_best_state()

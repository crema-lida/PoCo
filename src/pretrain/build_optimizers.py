import re
from torch.optim import AdamW
from transformers import get_scheduler

def roformer_layer_id(name: str, num_hidden_layers: int) -> int:
    """
    map parameter name to its layer id in RoFormer model.
      0            -> embeddings
      1..L         -> encoder.layer.{i}
      L+1          -> projection head
    """
    if name.startswith("roformer.embeddings"):
        return 0
    m = re.search(r"roformer\.encoder\.layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    return num_hidden_layers + 1  # pooler / projector / lm_head 等

def build_llrd_param_groups(model, base_lr, layer_decay, weight_decay):
    nl = model.config.num_hidden_layers
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = roformer_layer_id(n, nl)
        scale = layer_decay ** (nl + 1 - layer_id)
        # do not apply weight decay to biases and LayerNorm weights
        apply_wd = (p.ndim >= 2) and ("LayerNorm" not in n) and (not n.endswith(".bias"))
        wd = weight_decay if apply_wd else 0.0
        key = (scale, wd)
        groups.setdefault(key, []).append(p)
    
    optim_groups = [{"params": ps, "lr": base_lr * sc, "weight_decay": wd}
                    for (sc, wd), ps in groups.items()]
    return optim_groups


def build_optimizers(model, base_lr=3e-4, layer_decay=0.9, weight_decay=0.0,
                     scheduler_type="constant_with_warmup",
                     warmup_steps=0, max_steps=50000):
    optim_groups = build_llrd_param_groups(model, base_lr, layer_decay, weight_decay)
    optimizer = AdamW(optim_groups)
    lr_scheduler = get_scheduler(
        name=scheduler_type, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    return optimizer, lr_scheduler

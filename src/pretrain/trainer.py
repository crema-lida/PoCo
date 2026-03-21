import os
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel, TrainingArguments


class MomentumContrastTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        args: TrainingArguments = None,
        tau_pos: float = 0.1,
        tau_neg: float = 0.1,
        queue_size: int = 4096,
        momentum: float = 0.999,
        refine_start_step: int = 10_000,
        gram_teacher_initial_step: int = 10_000,
        gram_teacher_update_interval: int = 100_000,
        gram_lambda: float = 2.0,
        **kwargs
    ):
        super().__init__(model, args, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.queue_size = queue_size
        self.momentum = momentum

        self._queue = torch.randn((queue_size, self.model.proj_dim), dtype=self.model.dtype, device=self.model.device)
        self._queue = F.normalize(self._queue)
        self._queue_ptr = 0
        
        self.encoder_k = deepcopy(self.model)
        self.encoder_k.requires_grad_(False).eval().compile()
        self._last_step = 0

        self.gram_teacher = deepcopy(self.model.roformer)
        self.gram_teacher.requires_grad_(False).eval().compile()
        assert refine_start_step >= gram_teacher_initial_step
        self.refine_start_step = refine_start_step
        self.gram_teacher_inital_step = gram_teacher_initial_step
        self.gram_teacher_update_interval = gram_teacher_update_interval
        self.gram_lambda = gram_lambda

        self.losses = {'cl_loss': [], 'gram_loss': []}

    @torch.no_grad
    def _enqueue(self, features):
        B = features.size(0)
        Q = self.queue_size
        ptr = self._queue_ptr
        end = ptr + B
        if end <= Q:
            self._queue[ptr:end].copy_(features)
        else:
            first = Q - ptr
            self._queue[ptr:Q].copy_(features[:first])
            self._queue[0:end - Q].copy_(features[first:])
        self._queue_ptr = end % Q

    @torch.no_grad
    def _update_momentum_encoder(self):
        m = self.momentum
        for p_k, p in zip(self.encoder_k.parameters(), self.model.parameters()):
            p_k.data.mul_(m).add_(p.data, alpha=1.0 - m)

    def contrastive_loss(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        logits_pos = (anchor * pos).sum(dim=1, keepdim=True) / self.tau_pos
        logits_neg = anchor @ neg.T / self.tau_neg
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def gram_loss(self, h_s: torch.Tensor, h_t: torch.Tensor, mask: torch.Tensor):
        # hidden: (B, L, D); mask: (B, L)
        h_s = F.normalize(h_s, dim=-1)
        h_t = F.normalize(h_t, dim=-1)
        g_s = torch.bmm(h_s, h_s.transpose(1, 2))
        g_t = torch.bmm(h_t, h_t.transpose(1, 2))
        w = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).float()  # (B, L, L)
        diff2 = (g_s - g_t).pow(2) * w
        loss = diff2.sum() / w.sum()
        return loss
    
    def compute_loss(self, model: nn.Module, inputs: dict[str, list[torch.Tensor]], return_outputs=False, **kwargs):
        if self.state.global_step != self._last_step:
            self._update_momentum_encoder()
            self._last_step = self.state.global_step

            if self.state.global_step > self.refine_start_step + self.gram_teacher_update_interval \
                and self.state.global_step % self.gram_teacher_update_interval == 1 \
                or self.state.global_step == self.gram_teacher_inital_step:
                    self.gram_teacher.load_state_dict(self.model.roformer.state_dict())
                    print(f"=> Update gram teacher at step {self.state.global_step}")

        queries, keys = [], []
        hiddens = []

        for input_ids, attention_mask in zip(*inputs.values()):
            q = model(input_ids, attention_mask)
            queries.append(q['z'])
            hiddens.append(q['hidden'])

            with torch.no_grad():
                k = self.encoder_k(input_ids, attention_mask)
                keys.append(k['z'])

        neg = self._queue.clone()
        losses = []
        
        for i, k in enumerate(keys):
            for j, q in enumerate(queries):
                if i == j:
                    continue
                loss = self.contrastive_loss(q, k, neg)
                losses.append(loss)
        
        loss = torch.stack(losses).mean()

        # gram anchoring
        gram_loss = 0.
        n_terms = 0

        if self.state.global_step > self.refine_start_step:
            for input_ids, attention_mask, h_s in zip(*inputs.values(), hiddens):
                h_t = self.gram_teacher(input_ids, attention_mask).last_hidden_state
                gram_loss = gram_loss + self.gram_loss(h_s, h_t, attention_mask)
                n_terms += 1
            gram_loss = gram_loss / n_terms
            self.losses['gram_loss'].append(gram_loss.item())
        
        self.losses['cl_loss'].append(loss.item())

        loss = loss + self.gram_lambda * gram_loss

        # Update queue
        keys = torch.cat(keys)
        keys = concat_all_gather(keys)
        self._enqueue(keys)

        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for name, losses in self.losses.items():
            if len(losses) == 0:
                continue
            logs[f'train/{name}'] = sum(losses) / len(losses)
            losses.clear()
        super().log(logs, start_time)
    
    def _save(self, output_dir: str | None = None, state_dict=None):
        super()._save(output_dir, state_dict)
        moco_state = {
            'encoder_k': self.encoder_k.state_dict(),
            'gram_teacher': self.gram_teacher.state_dict(),
            'queue': self._queue,
            'queue_ptr': self._queue_ptr,
            'last_step': self._last_step,
        }
        torch.save(moco_state, os.path.join(output_dir, 'moco_state.pt'))

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        moco_state_path = os.path.join(resume_from_checkpoint, 'moco_state.pt')
        moco_state: dict = torch.load(moco_state_path)
        
        self.encoder_k.load_state_dict(moco_state['encoder_k'])
        self.gram_teacher.load_state_dict(moco_state['gram_teacher'])
        self._queue.copy_(moco_state['queue'])
        self._queue_ptr = moco_state['queue_ptr']
        self._last_step = moco_state['last_step']


@torch.no_grad
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.

    ***Warning***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

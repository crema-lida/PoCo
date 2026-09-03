import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler


class Transform(nn.Module):
    def __init__(self, data=None, log10_idx=None, logm1_idx=None, num_features=1):
        super().__init__()
        float64 = torch.float64
        if data is None:
            self.center = nn.Buffer(torch.zeros(num_features, dtype=float64))
            self.scale = nn.Buffer(torch.ones(num_features, dtype=float64))
            self.log10_idx = nn.Buffer(torch.zeros(num_features, dtype=torch.bool))
            self.logm1_idx = nn.Buffer(torch.zeros(num_features, dtype=torch.bool))
        else:            
            data = torch.as_tensor(np.array(data, copy=True), dtype=float64)
            if log10_idx is None:
                log10_idx = torch.zeros(data.shape[-1], dtype=torch.bool)
            if logm1_idx is None:
                logm1_idx = torch.zeros(data.shape[-1], dtype=torch.bool)
            data[:, log10_idx] = torch.log10(data[:, log10_idx])
            data[:, logm1_idx] = torch.log10(data[:, logm1_idx] - 1)
            
            scaler = RobustScaler().fit(data)
            self.center = nn.Buffer(torch.as_tensor(scaler.center_, dtype=float64))
            self.scale = nn.Buffer(torch.as_tensor(scaler.scale_, dtype=float64))
            self.log10_idx = nn.Buffer(torch.as_tensor(log10_idx, dtype=torch.bool))
            self.logm1_idx = nn.Buffer(torch.as_tensor(logm1_idx, dtype=torch.bool))

    def forward(self, x: torch.Tensor, inverse=False):
        s_log10 = np.s_[:, self.log10_idx]
        s_logm1 = np.s_[:, self.logm1_idx]
        if inverse:
            x = x * self.scale + self.center
            x[s_logm1] = 10 ** x[s_logm1] + 1
            x[s_log10] = 10 ** x[s_log10]
        else:
            x = x.clone()
            x[s_log10] = torch.log10(x[s_log10])
            x[s_logm1] = torch.log10(x[s_logm1] - 1)
            x = (x - self.center) / self.scale
        return x

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp


class GCNNet(torch.nn.Module):
    def __init__(
        self,
        num_features_xd: int = 62,
        num_features_edge: int = 10,
        n_output: int = 1,
        output_dim: int = 64,
        output_dim2: int = 16,
        dropout: float = 0.2,
        num_net: int = 5,
        file=None,
    ):
        super().__init__()
        self.num_net = num_net
        self.drug1_gcn1 = nn.ModuleList(
            [GCNConv(num_features_xd, num_features_xd) for _ in range(num_net)]
        )
        self.drug1_gcn2 = nn.ModuleList(
            [GCNConv(num_features_xd, num_features_xd) for _ in range(num_net)]
        )
        self.drug1_gcn3 = nn.ModuleList(
            [GCNConv(num_features_xd, num_features_xd) for _ in range(num_net)]
        )
        self.drug1_gcn4 = nn.ModuleList(
            [GCNConv(num_features_xd, num_features_xd) for _ in range(num_net)]
        )
        self.drug1_gcn5 = nn.ModuleList(
            [GCNConv(num_features_xd, num_features_xd) for _ in range(num_net)]
        )

        self.drug2_gcn1 = nn.ModuleList(
            [GCNConv(num_features_edge, num_features_edge) for _ in range(num_net)]
        )
        self.drug2_gcn2 = nn.ModuleList(
            [GCNConv(num_features_edge, num_features_edge) for _ in range(num_net)]
        )
        self.drug2_gcn3 = nn.ModuleList(
            [GCNConv(num_features_edge, num_features_edge) for _ in range(num_net)]
        )
        self.drug2_gcn4 = nn.ModuleList(
            [GCNConv(num_features_edge, num_features_edge) for _ in range(num_net)]
        )
        self.drug2_gcn5 = nn.ModuleList(
            [GCNConv(num_features_edge, num_features_edge) for _ in range(num_net)]
        )

        self.drug1_fc_g1 = nn.ModuleList(
            [nn.Linear(num_features_xd * 2, output_dim) for _ in range(num_net)]
        )
        self.drug2_fc_g1 = nn.ModuleList(
            [nn.Linear(num_features_edge * 2, output_dim2) for _ in range(num_net)]
        )

        self.fc1 = nn.Linear(output_dim * num_net + output_dim2 * num_net, 64)
        self.fc2 = nn.Linear(output_dim2 * num_net, 64)
        self.out = nn.Linear(64, n_output)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(self, data_data, data_data2, device):
        x11 = torch.Tensor().to(device)
        x22 = torch.Tensor().to(device)
        for module, data, data2 in zip(
            zip(
                self.drug1_gcn1,
                self.drug1_gcn2,
                self.drug1_gcn3,
                self.drug1_gcn4,
                self.drug1_gcn5,
                self.drug2_gcn1,
                self.drug2_gcn2,
                self.drug2_gcn3,
                self.drug2_gcn4,
                self.drug2_gcn5,
                self.drug1_fc_g1,
                self.drug2_fc_g1,
            ),
            data_data,
            data_data2,
        ):
            data = data.to(device)
            data2 = data2.to(device)
            x1, edge_index1, batch = data.x, data.edge_index, data.batch
            x2, edge_index2, edge_size = data2.x, data2.edge_index, data2.batch

            drug1_gcn1 = module[0]
            drug1_gcn2 = module[1]
            drug1_gcn3 = module[2]
            drug1_gcn4 = module[3]
            drug1_gcn5 = module[4]

            drug2_gcn1 = module[5]
            drug2_gcn2 = module[6]
            drug2_gcn3 = module[7]
            drug2_gcn4 = module[8]
            drug2_gcn5 = module[9]

            drug1_fc_g1 = module[10]
            drug2_fc_g1 = module[11]

            x1 = drug1_gcn1(x1, edge_index1)
            x1 = self.relu(x1)
            x1 = drug1_gcn2(x1, edge_index1)
            x1 = self.relu(x1)
            x1 = drug1_gcn3(x1, edge_index1)
            x1 = torch.cat((data.x, x1), 1)

            x2 = drug2_gcn1(x2, edge_index2)
            x2 = self.relu(x2)
            x2 = drug2_gcn2(x2, edge_index2)
            x2 = self.relu(x2)
            x2 = drug2_gcn3(x2, edge_index2)
            x2 = torch.cat((data2.x, x2), 1)

            x1 = gmp(x1, batch)
            x1 = drug1_fc_g1(x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)
            x11 = torch.cat((x11, x1), 1)

            x2 = gmp(x2, edge_size)
            x2 = drug2_fc_g1(x2)
            x2 = self.relu(x2)
            x2 = self.dropout(x2)
            x22 = torch.cat((x22, x2), 1)

        xx = torch.cat((x11, x22), 1)
        xc = self.fc1(xx)
        return xc

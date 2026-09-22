import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    """Per-channel shape function: 3 x (conv, BN, ReLU), a final conv, then GAP.

    BA adaptation (F2): `dropout_p` inserts `nn.Dropout` after each ReLU inside
    `conv_pool`. It defaults to 0.0 and the module is then NOT INSERTED AT ALL,
    rather than added with p = 0. That distinction is the whole point: a
    zero-probability dropout still sits in the module list, so the state dict
    keys, the module order and (depending on the backend) the RNG stream stop
    matching the published network, and every previously reported number would
    have to be re-run to stay comparable. With the module omitted, `dropout_p:
    0.0` is the published architecture exactly -- asserted by
    `scripts/check_dropout_identity.py`.

    Plain `nn.Dropout`, not `nn.Dropout1d`: dropping whole feature maps is the
    stronger regulariser and a defensible alternative, but "add dropout" in its
    usual sense means the element-wise one, and the point of F2 is to test the
    standard remedy rather than a variant of it.
    """

    def __init__(
        self,
        in_channel=1,
        hidden_channel=20,
        kernel_size=29,
        dropout_p=0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        assert 0.0 <= dropout_p < 1.0, f"dropout_p out of range: {dropout_p}"
        padding = (kernel_size - 1) // 2
        self.dropout_p = float(dropout_p)

        layers = []
        for block in range(3):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channel if block == 0 else hidden_channel,
                    out_channels=hidden_channel,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm1d(hidden_channel))
            layers.append(nn.ReLU())
            if self.dropout_p > 0.0:
                layers.append(nn.Dropout(self.dropout_p))
        self.conv_pool = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.lin_comb = nn.Linear(hidden_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        downpool = torch.mean(last_act, dim=2)
        return self.sigmoid(self.lin_comb(downpool))

    def forward_logit(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        downpool = torch.mean(last_act, dim=2)
        return self.lin_comb(downpool)

    def class_act(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        last_act = last_act.transpose(1, 2)
        class_act = self.lin_comb(last_act)
        return class_act.transpose(1, 2)

    def only_conv(self, sig):
        temp = self.conv_pool(sig)
        last_act = self.final_conv(temp)
        return last_act


class NeuralAdditiveModel(nn.Module):
    def __init__(self, module_list):
        super().__init__()

        self.module_list = nn.ModuleList(module_list)
        self.multiplier_list = nn.ParameterList(
            [nn.Parameter(torch.Tensor([1.0])) for _mod in module_list]
        )
        self.bias = nn.Parameter(torch.Tensor([0.0]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ls):
        assert len(x_ls) == len(self.module_list)
        stacked = torch.stack(
            [
                multiplier * module.forward_logit(x)
                for module, multiplier, x in zip(
                    self.module_list, self.multiplier_list, x_ls
                )
            ]
        )
        return self.sigmoid(stacked.sum(dim=0) + self.bias)

    def compute_logits(self, x_ls):
        assert len(x_ls) == len(self.module_list)
        logit_ls = [
            multiplier * module.forward_logit(x)
            for module, multiplier, x in zip(
                self.module_list, self.multiplier_list, x_ls
            )
        ]
        return logit_ls, self.bias


class MLPCombiner(nn.Module):
    def __init__(self, module_list, mlp_classifier):
        super().__init__()

        self.module_list = nn.ModuleList(module_list)
        self.mlp_classifier = mlp_classifier

    def forward(self, x_ls):
        assert len(x_ls) == len(self.module_list)
        catted = torch.cat(
            [
                torch.mean(module.only_conv(x), dim=2)
                for module, x in zip(self.module_list, x_ls)
            ],
            dim=1,
        )
        return self.mlp_classifier(catted)


class MLPClassifier(nn.Module):
    def __init__(self, meta_size, intermediate_size):
        super().__init__()
        self.meta_size = meta_size
        self.intermediate_size = intermediate_size
        self.meta_net = nn.Sequential(
            nn.Linear(self.meta_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, meta):
        return self.sigmoid(self.meta_net(meta))

    def forward_logit(self, meta):
        return self.meta_net(meta)


class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, meta):
        return self.sigmoid(self.layer(meta))

    def forward_logit(self, meta):
        return self.meta_net(meta)

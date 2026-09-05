import torch
import torch.optim as optim

import util
from model import AMGTSA


class AMGTSATrainer:
    def __init__(
        self,
        scaler,
        in_dim,
        seq_length,
        num_nodes,
        hidden_dim,
        dropout,
        learning_rate,
        weight_decay,
        device,
        supports,
        huber_delta=1.0,
        use_asg=True,
        use_adg=True,
        use_tasatt=True,
        use_dccn=True,
    ):
        self.model = AMGTSA(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            in_dim=in_dim,
            out_dim=seq_length,
            residual_channels=hidden_dim,
            dilation_channels=hidden_dim,
            skip_channels=hidden_dim * 8,
            end_channels=hidden_dim * 16,
            use_asg=use_asg,
            use_adg=use_adg,
            use_tasatt=use_tasatt,
            use_dccn=use_dccn,
        ).to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scaler = scaler
        self.huber_delta = huber_delta
        self.clip = 5.0

    def train(self, input_data, real_value):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input_data).transpose(1, 3)
        real = torch.unsqueeze(real_value, dim=1)
        prediction = self.scaler.inverse_transform(output)

        loss = util.masked_huber(
            prediction, real, null_val=0.0, delta=self.huber_delta
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.masked_mape(prediction, real, 0.0).item()
        rmse = util.masked_rmse(prediction, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input_data, real_value):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data).transpose(1, 3)
            real = torch.unsqueeze(real_value, dim=1)
            prediction = self.scaler.inverse_transform(output)
            loss = util.masked_huber(
                prediction, real, null_val=0.0, delta=self.huber_delta
            )
            mape = util.masked_mape(prediction, real, 0.0).item()
            rmse = util.masked_rmse(prediction, real, 0.0).item()
        return loss.item(), mape, rmse

trainer = AMGTSATrainer

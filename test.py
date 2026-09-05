import argparse

import numpy as np
import torch

import util
from model import AMGTSA


parser = argparse.ArgumentParser(description="Evaluate AMG-TSA")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data", type=str, default="METR-LA")
parser.add_argument("--adjdata", type=str, default="")
parser.add_argument("--adjtype", type=str, default="doubletransition")
parser.add_argument("--seq_length", type=int, default=12)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--in_dim", type=int, default=2)
parser.add_argument("--num_nodes", type=int, default=207)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj_mx]

    model = AMGTSA(
        device=device,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        supports=supports,
        in_dim=args.in_dim,
        out_dim=args.seq_length,
        residual_channels=args.hidden_dim,
        dilation_channels=args.hidden_dim,
        skip_channels=args.hidden_dim * 8,
        end_channels=args.hidden_dim * 16,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("AMG-TSA model loaded successfully")

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]
    real_y = torch.tensor(dataloader["y_test"], dtype=torch.float32, device=device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]

    outputs = []
    for x, _ in dataloader["test_loader"].get_iterator():
        test_x = torch.tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
        with torch.no_grad():
            outputs.append(model(test_x).transpose(1, 3).squeeze())

    y_hat = torch.cat(outputs, dim=0)[: real_y.size(0)]
    mae_list, mape_list, rmse_list = [], [], []
    for horizon in range(args.seq_length):
        pred = scaler.inverse_transform(y_hat[:, :, horizon])
        real = real_y[:, :, horizon]
        mae, mape, rmse = util.metric(pred, real)
        print(
            f"Horizon {horizon + 1:02d}: Test MAE={mae:.4f}, "
            f"Test MAPE={mape:.4f}, Test RMSE={rmse:.4f}"
        )
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    print(
        f"Average over {args.seq_length} horizons: Test MAE={np.mean(mae_list):.4f}, "
        f"Test MAPE={np.mean(mape_list):.4f}, Test RMSE={np.mean(rmse_list):.4f}"
    )


if __name__ == "__main__":
    main()

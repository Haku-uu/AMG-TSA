import argparse
import random
import time

import numpy as np
import torch

import util
from engine import AMGTSATrainer


parser = argparse.ArgumentParser(description="Train AMG-TSA")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data", type=str, default="PEMS_BAY")
parser.add_argument("--adjdata", type=str, default="")
parser.add_argument("--adjtype", type=str, default="doubletransition")
parser.add_argument("--seq_length", type=int, default=12)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--in_dim", type=int, default=2)
parser.add_argument("--num_nodes", type=int, default=325)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--huber_delta", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save", type=str, default="")
parser.add_argument("--expid", type=int, default=1)


parser.add_argument("--without_dccn", action="store_true")
parser.add_argument("--without_asg", action="store_true")
parser.add_argument("--without_adg", action="store_true")
parser.add_argument("--without_tasatt", action="store_true")

args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(args.seed)
    device = torch.device(args.device)

    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj_mx]
    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    engine = AMGTSATrainer(
        scaler=scaler,
        in_dim=args.in_dim,
        seq_length=args.seq_length,
        num_nodes=args.num_nodes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        supports=supports,
        huber_delta=args.huber_delta,
        use_asg=not args.without_asg,
        use_adg=not args.without_adg,
        use_tasatt=not args.without_tasatt,
        use_dccn=not args.without_dccn,
    )

    print("Start training AMG-TSA...", flush=True)
    history = []
    train_times, val_times = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []
        start = time.time()
        dataloader["train_loader"].shuffle()

        for iteration, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            train_x = torch.tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
            train_y = torch.tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
            metrics = engine.train(train_x, train_y[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])

            if iteration % args.print_every == 0:
                print(
                    f"Iter: {iteration:03d}, Train Huber: {metrics[0]:.4f}, "
                    f"Train MAPE: {metrics[1]:.4f}, Train RMSE: {metrics[2]:.4f}",
                    flush=True,
                )

        train_times.append(time.time() - start)

        val_loss, val_mape, val_rmse = [], [], []
        val_start = time.time()
        for x, y in dataloader["val_loader"].get_iterator():
            val_x = torch.tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
            val_y = torch.tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
            metrics = engine.eval(val_x, val_y[:, 0, :, :])
            val_loss.append(metrics[0])
            val_mape.append(metrics[1])
            val_rmse.append(metrics[2])
        val_times.append(time.time() - val_start)

        mean_train = np.mean(train_loss)
        mean_val = np.mean(val_loss)
        history.append(mean_val)
        print(
            f"Epoch: {epoch:03d}, Train Huber: {mean_train:.4f}, "
            f"Train MAPE: {np.mean(train_mape):.4f}, Train RMSE: {np.mean(train_rmse):.4f}, "
            f"Valid Huber: {mean_val:.4f}, Valid MAPE: {np.mean(val_mape):.4f}, "
            f"Valid RMSE: {np.mean(val_rmse):.4f}, Training Time: {train_times[-1]:.4f}/epoch",
            flush=True,
        )
        torch.save(
            engine.model.state_dict(),
            f"{args.save}_epoch_{epoch}_{round(mean_val, 4)}.pth",
        )

    best_epoch = int(np.argmin(history)) + 1
    best_loss = history[best_epoch - 1]
    engine.model.load_state_dict(
        torch.load(
            f"{args.save}_epoch_{best_epoch}_{round(best_loss, 4)}.pth",
            map_location=device,
        )
    )

    outputs = []
    real_y = torch.tensor(dataloader["y_test"], dtype=torch.float32, device=device)
    real_y = real_y.transpose(1, 3)[:, 0, :, :]
    for x, _ in dataloader["test_loader"].get_iterator():
        test_x = torch.tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
        with torch.no_grad():
            outputs.append(engine.model(test_x).transpose(1, 3).squeeze())

    y_hat = torch.cat(outputs, dim=0)[: real_y.size(0)]
    mae_list, mape_list, rmse_list = [], [], []
    for horizon in range(args.seq_length):
        pred = scaler.inverse_transform(y_hat[:, :, horizon])
        real = real_y[:, :, horizon]
        mae, mape, rmse = util.metric(pred, real)
        print(
            f"Horizon {horizon + 1:02d}: MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}"
        )
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

    print(
        f"Average over {args.seq_length} horizons: MAE={np.mean(mae_list):.4f}, "
        f"MAPE={np.mean(mape_list):.4f}, RMSE={np.mean(rmse_list):.4f}"
    )
    print(f"Average Training Time: {np.mean(train_times):.4f} secs/epoch")
    print(f"Average Validation Time: {np.mean(val_times):.4f} secs")

    torch.save(
        engine.model.state_dict(),
        f"{args.save}_exp{args.expid}_best_{round(best_loss, 4)}.pth",
    )


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time spent: {time.time() - start:.4f} secs")

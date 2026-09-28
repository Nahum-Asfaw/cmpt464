"""Train each assignment shape, then evaluate and export its reconstruction."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import ShapeDataset
from evaluation import evaluate
from losses import bce_loss, eikonal_loss, mse_loss
from models import IMNet

REPO_DIR = Path(__file__).resolve().parents[1]
DEVICE = "cuda"
SEED = 0
HIDDEN_DIMS = [256, 256, 256, 256]


def train(
    model: IMNet,
    loader: DataLoader,
    optimizer: Adam,
    loss_fn,
    steps: int,
    output_dir: Path,
    shape: str,
    *,
    z: torch.Tensor | None = None,
    eikonal_weight: float | None = None,
) -> None:
    """Fit one shape and write its per-step CSV and text log.

    steps counts optimizer updates, including short batches at epoch ends.
    The loader reuses fixed points and shuffles their order each epoch.
    eikonal_weight=None disables the penalty; zero still computes and logs it.
    """
    if eikonal_weight is not None:
        eikonal_generator = torch.Generator(device=DEVICE).manual_seed(100)

    log_path = output_dir / f"{shape}_train.log"
    loss_path = output_dir / f"{shape}_loss.csv"
    step = 0
    epoch = 1

    with (
        log_path.open("w", encoding="utf-8") as log_file,
        loss_path.open("w", newline="", encoding="utf-8") as loss_file,
    ):
        writer = csv.writer(loss_file)
        writer.writerow(["step", "loss", "data_loss", "eikonal_loss"])

        while step < steps:
            for points, targets in loader:
                # The model's leading dimension counts shapes; this batch has one.
                points = points.unsqueeze(0)
                targets = targets.unsqueeze(0)

                optimizer.zero_grad()
                prediction = model(points, z)
                data_loss = loss_fn(prediction, targets)
                loss = data_loss
                eikonal_value = ""

                if eikonal_weight is not None:
                    # Fresh uniform queries use a separate RNG from the fixed supervision.
                    eikonal_points = torch.rand(
                        1, loader.batch_size, 3,
                        device=DEVICE, generator=eikonal_generator,
                    ) - 0.5
                    eikonal_points.requires_grad_(True)
                    penalty = eikonal_loss(model(eikonal_points), eikonal_points)
                    loss = data_loss + eikonal_weight * penalty
                    eikonal_value = penalty.item()

                loss.backward()
                optimizer.step()
                step += 1

                loss_value = loss.item()
                data_value = data_loss.item()
                writer.writerow([step, loss_value, data_value, eikonal_value])
                message = (
                    f"{shape} epoch={epoch} step={step}/{steps} "
                    f"loss={loss_value:.6g} data_loss={data_value:.6g} "
                    f"eikonal_loss={eikonal_value}"
                )
                print(message)
                log_file.write(message + "\n")

                if step == steps:
                    break
            epoch += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the six assignment shapes.")
    parser.add_argument(
        "-p",
        dest="problem",
        choices=["prob1", "prob2_1", "prob2_2", "prob2_3", "prob2_4", "prob3"],
        required=True,
        help="Assignment problem to run.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--steps", type=int, default=5000, help="Updates per shape.")
    parser.add_argument("--bs", type=int, default=4096, help="Query points per batch.")
    parser.add_argument("--num_points", type=int, default=32768, help="Fixed training points per shape.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Gaussian noise std for prob2_4 only.")
    parser.add_argument(
        "--eikonal_weight",
        type=float,
        default=1e-5,
        help="Eikonal loss weight for prob2_3 only.",
    )
    args = parser.parse_args()

    # Start with occupancy regression; each problem changes the relevant choices.
    target = "occ"
    output_format = "imnet"
    sampling = "uniform"
    sigma = 0.02
    loss_fn = mse_loss
    eikonal_weight = None

    if args.problem == "prob2_1":
        output_format = "logit"
        loss_fn = bce_loss
    elif args.problem in ["prob2_2", "prob2_3", "prob2_4"]:
        target = "sdf"
        output_format = "logit"

    if args.problem == "prob2_3":
        eikonal_weight = args.eikonal_weight
    if args.problem == "prob2_4":
        sampling = "near_surface"
        sigma = args.sigma
    elif args.problem == "prob3":
        sampling = "near_surface"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_DIR / "output" / args.problem / timestamp
    output_dir.mkdir(parents=True)

    for shape in ["chair", "teapot", "plane", "spot", "blub", "bob"]:
        mesh_path = REPO_DIR / "data" / f"{shape}.obj"
        # CUDA work is asynchronous; synchronize at both ends of the timer.
        torch.cuda.synchronize()
        start = perf_counter()
        # Restart initialization from the same seed for each shape.
        torch.manual_seed(SEED)

        z = None
        if args.problem == "prob3":
            model = IMNet([1024, 1024, 1024, 512, 256, 128], z_dim=256, output_format="imnet").to(DEVICE)
            weights_path = REPO_DIR / "weights" / "imnet_decoder.pt"
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
            # Freeze decoder parameters while preserving the gradient path to z.
            # The provided eval() call changes module mode; it does not freeze weights.
            ##### your code starts here ##### Problem 3 (Bonus): freeze decoder
            raise NotImplementedError("Problem 3: freeze decoder parameters")
            ##### your code ends here #####
            model.eval()
            z = torch.zeros(1, 256, device=DEVICE, requires_grad=True)
            # Set parameters to the iterable used by Adam below. Only z is optimized.
            ##### your code starts here ##### Problem 3 (Bonus): optimizer parameters
            raise NotImplementedError("Problem 3: select optimizer parameters")
            ##### your code ends here #####
        else:
            model = IMNet(HIDDEN_DIMS, output_format=output_format).to(DEVICE)
            model.train()
            parameters = model.parameters()

        optimizer = Adam(parameters, lr=args.lr)
        dataset = ShapeDataset(
            str(mesh_path),
            args.num_points,
            target=target,
            sampling=sampling,
            sigma=sigma,
            seed=SEED,
            device=DEVICE,
        )
        # The dataset already lives on CUDA; batching stays in this process.
        loader = DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=torch.Generator().manual_seed(SEED),
        )
        train(
            model,
            loader,
            optimizer,
            loss_fn,
            args.steps,
            output_dir,
            shape,
            z=z,
            eikonal_weight=eikonal_weight,
        )
        torch.cuda.synchronize()
        training_seconds = perf_counter() - start

        if z is not None:
            torch.save(z.detach().cpu(), output_dir / f"{shape}_z.pt")

        start = perf_counter()
        metrics = evaluate(model, z, mesh_path, target, args.bs, output_dir)
        torch.cuda.synchronize()
        metrics["training_seconds"] = training_seconds
        metrics["evaluation_seconds"] = perf_counter() - start
        with (output_dir / f"{shape}_metrics.json").open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        message = f"{shape}: {json.dumps(metrics)}"
        print(message)
        with (output_dir / f"{shape}_train.log").open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")


if __name__ == "__main__":
    main()

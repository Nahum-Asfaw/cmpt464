"""Command line entry: python src/main.py -p prob1 | prob2 | prob3"""

import argparse
import json
import warnings
from datetime import datetime

# kaolin warns at import time that its Jupyter viewer packages are missing; this assignment does not use them
warnings.filterwarnings("ignore", message="Error importing kaolin.visualize.ipython")

import torch
from data import EVAL_SHAPES, REPO_DIR, TRAIN_SHAPES, G, load_mesh, sdf_grid, write_obj
from dmc import cube_grid, dual_marching_cubes
from dual_contouring import crossed_edges, dual_contouring, estimate_hermite, exact_hermite
from flexicubes import fit, signed_distance
from metrics import evaluate
from neural_dual_contouring import predict_mesh, train

OUTPUT_DIR = REPO_DIR / "output"
SEED = 0


def print_metrics(metrics):
    """Print the metrics as a table: one row per shape and method ("-" where there are no sharp edges).

    metrics  dict   metrics[shape][method] = result of evaluate (or fit)
    """
    print(f"{'shape':<10} {'method':<10} {'chamfer':>8} {'edge_ch':>8} {'#verts':>7} {'#faces':>7}")
    for name, methods in metrics.items():
        for method, result in methods.items():
            edge_chamfer = "-" if result["edge_chamfer"] is None else f"{result['edge_chamfer']:.4f}"
            print(f"{name:<10} {method:<10} {result['chamfer']:>8.4f} {edge_chamfer:>8} " f"{result['num_vertices']:>7} {result['num_faces']:>7}")


def run_prob1(output_dir):
    """Problem 1: dual contouring with exact vs estimated Hermite data, on every evaluation shape.

    output_dir  Path   where the OBJ files and metrics.json are written
    """
    metrics = {}
    for name in EVAL_SHAPES:
        vertices, faces = load_mesh(name)

        has_hermite, exact_t, exact_normal = exact_hermite(vertices, faces)
        sdf = sdf_grid(vertices, faces, has_hermite)
        exact_vertices, exact_faces = dual_contouring(sdf, has_hermite, exact_t, exact_normal)

        estimated_t, estimated_normal = estimate_hermite(sdf)
        inside, sign_change, has_vertex = crossed_edges(sdf)
        estimated_vertices, estimated_faces = dual_contouring(sdf, sign_change, estimated_t, estimated_normal)

        write_obj(output_dir / f"{name}_input.obj", vertices, faces)
        write_obj(output_dir / f"{name}_exact.obj", exact_vertices, exact_faces)
        write_obj(output_dir / f"{name}_estimated.obj", estimated_vertices, estimated_faces)

        metrics[name] = {
            "exact": evaluate(exact_vertices, exact_faces, vertices, faces),
            "estimated": evaluate(estimated_vertices, estimated_faces, vertices, faces),
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print_metrics(metrics)


def run_prob2(output_dir):
    """Problem 2: NDC trained on the training shapes vs estimated-Hermite dual contouring, on every evaluation shape.

    output_dir  Path   where the model, the training log, the OBJ files and metrics.json are written
    """
    model = train(TRAIN_SHAPES, output_dir)

    metrics = {}
    for name in EVAL_SHAPES:
        vertices, faces = load_mesh(name)

        has_hermite, exact_t, exact_normal = exact_hermite(vertices, faces)
        sdf = sdf_grid(vertices, faces, has_hermite)

        neural_vertices, neural_faces = predict_mesh(model, sdf)

        estimated_t, estimated_normal = estimate_hermite(sdf)
        inside, sign_change, has_vertex = crossed_edges(sdf)
        estimated_vertices, estimated_faces = dual_contouring(sdf, sign_change, estimated_t, estimated_normal)

        write_obj(output_dir / f"{name}_input.obj", vertices, faces)
        write_obj(output_dir / f"{name}_neural.obj", neural_vertices, neural_faces)
        write_obj(output_dir / f"{name}_estimated.obj", estimated_vertices, estimated_faces)

        metrics[name] = {
            "neural": evaluate(neural_vertices, neural_faces, vertices, faces),
            "estimated": evaluate(estimated_vertices, estimated_faces, vertices, faces),
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print_metrics(metrics)


def run_prob3(output_dir):
    """Problem 3: DMC extracted from the SDF, and FlexiCubes fitted from a random SDF, on every evaluation shape.

    output_dir  Path   where the fitting log, the OBJ files and metrics.json are written
    """
    metrics = {}
    for name in EVAL_SHAPES:
        vertices, faces = load_mesh(name)

        # SDF at every grid point, computed directly from the mesh (DMC needs no Hermite data)
        positions = cube_grid()[0]
        sdf = signed_distance(positions, vertices, faces).reshape(G + 1, G + 1, G + 1)
        dmc_vertices, dmc_faces = dual_marching_cubes(sdf, positions)

        write_obj(output_dir / f"{name}_dmc.obj", dmc_vertices, dmc_faces)

        metrics[name] = {"dmc": evaluate(dmc_vertices, dmc_faces, vertices, faces), "flexicubes": fit(name, output_dir)}

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print_metrics(metrics)


def main():
    """Parse the problem name, seed torch, and run that problem into a new time-stamped output folder."""
    runs = {"prob1": run_prob1, "prob2": run_prob2, "prob3": run_prob3}
    parser = argparse.ArgumentParser(description="CMPT 464/764 Assignment 2: Neural Dual Contouring")
    parser.add_argument("-p", "--problem", required=True, choices=list(runs))
    arguments = parser.parse_args()

    torch.manual_seed(SEED)
    output_dir = OUTPUT_DIR / arguments.problem / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    runs[arguments.problem](output_dir)


if __name__ == "__main__":
    main()

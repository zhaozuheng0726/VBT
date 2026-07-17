#!/usr/bin/env python3
"""Compare two OBJ surfaces with deterministic area-weighted sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                polygon = [int(field.split("/", 1)[0]) for field in line.split()[1:]]
                polygon = [index - 1 if index > 0 else len(vertices) + index for index in polygon]
                for offset in range(1, len(polygon) - 1):
                    triangles.append([polygon[0], polygon[offset], polygon[offset + 1]])

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError(f"{path}: no valid OBJ vertices")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError(f"{path}: no valid OBJ faces")
    if np.min(triangle_array) < 0 or np.max(triangle_array) >= len(vertex_array):
        raise ValueError(f"{path}: face index outside vertex array")
    return vertex_array, triangle_array


def triangle_geometry(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = vertices[triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("mesh has no non-degenerate triangles")
    points = points[valid]
    double_area = double_area[valid]
    normals = cross[valid] / double_area[:, None]
    return points, 0.5 * double_area, normals


def sample_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    points, areas, normals = triangle_geometry(vertices, triangles)
    total_area = float(np.sum(areas, dtype=np.float64))
    face_indices = rng.choice(len(points), size=count, p=areas / total_area)
    selected = points[face_indices]

    uv = rng.random((count, 2))
    reflected = np.sum(uv, axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    samples = (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    )
    return samples, normals[face_indices], total_area


def make_query_mesh(vertices: np.ndarray, triangles: np.ndarray) -> trimesh.Trimesh:
    points = vertices[triangles]
    double_area = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    valid = double_area > np.finfo(np.float64).eps
    return trimesh.Trimesh(vertices=vertices, faces=triangles[valid], process=False)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def component_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    _, areas, _ = triangle_geometry(vertices, triangles)
    points = vertices[triangles]
    double_area = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    valid = double_area > np.finfo(np.float64).eps
    valid_triangles = triangles[valid]

    sets = DisjointSet(len(vertices))
    for first, second, third in valid_triangles:
        sets.union(int(first), int(second))
        sets.union(int(first), int(third))

    totals: dict[int, float] = {}
    for triangle, area in zip(valid_triangles, areas, strict=True):
        root = sets.find(int(triangle[0]))
        totals[root] = totals.get(root, 0.0) + float(area)
    return np.sort(np.fromiter(totals.values(), dtype=np.float64))[::-1]


def direction_metrics(
    source_points: np.ndarray,
    source_normals: np.ndarray,
    target_mesh: trimesh.Trimesh,
    chunk_size: int = 25_000,
) -> dict[str, float]:
    distance_chunks: list[np.ndarray] = []
    normal_dot_chunks: list[np.ndarray] = []
    target_normals = np.asarray(target_mesh.face_normals, dtype=np.float64)
    for start in range(0, len(source_points), chunk_size):
        stop = min(start + chunk_size, len(source_points))
        _, distances, triangle_indices = trimesh.proximity.closest_point(
            target_mesh, source_points[start:stop]
        )
        distance_chunks.append(distances)
        normal_dot_chunks.append(
            np.sum(source_normals[start:stop] * target_normals[triangle_indices], axis=1)
        )

    distances = np.concatenate(distance_chunks)
    normal_dot = np.concatenate(normal_dot_chunks)
    normal_dot = np.clip(normal_dot, -1.0, 1.0)
    oriented_angles = np.degrees(np.arccos(normal_dot))
    unoriented_angles = np.degrees(np.arccos(np.abs(normal_dot)))
    return {
        "mean_distance": float(np.mean(distances, dtype=np.float64)),
        "mean_squared_distance": float(np.mean(np.square(distances), dtype=np.float64)),
        "p95_distance": float(np.percentile(distances, 95.0)),
        "p99_distance": float(np.percentile(distances, 99.0)),
        "max_distance": float(np.max(distances)),
        "mean_normal_angle_degrees": float(np.mean(oriented_angles, dtype=np.float64)),
        "p95_normal_angle_degrees": float(np.percentile(oriented_angles, 95.0)),
        "mean_unoriented_normal_angle_degrees": float(
            np.mean(unoriented_angles, dtype=np.float64)
        ),
        "p95_unoriented_normal_angle_degrees": float(
            np.percentile(unoriented_angles, 95.0)
        ),
    }


def mesh_summary(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, object]:
    areas = component_areas(vertices, triangles)
    total_area = float(np.sum(areas, dtype=np.float64))
    relative_threshold = total_area * 1.0e-4
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "surface_area": total_area,
        "connected_components": int(len(areas)),
        "components_above_0_01_percent_area": int(np.count_nonzero(areas >= relative_threshold)),
        "largest_component_area_fraction": float(areas[0] / total_area),
        "top_component_areas": [float(value) for value in areas[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    reference_vertices, reference_triangles = load_obj(args.reference)
    candidate_vertices, candidate_triangles = load_obj(args.candidate)
    rng = np.random.default_rng(args.seed)
    reference_points, reference_normals, _ = sample_surface(
        reference_vertices, reference_triangles, args.samples, rng
    )
    candidate_points, candidate_normals, _ = sample_surface(
        candidate_vertices, candidate_triangles, args.samples, rng
    )
    reference_mesh = make_query_mesh(reference_vertices, reference_triangles)
    candidate_mesh = make_query_mesh(candidate_vertices, candidate_triangles)

    reference_to_candidate = direction_metrics(
        reference_points, reference_normals, candidate_mesh
    )
    candidate_to_reference = direction_metrics(
        candidate_points, candidate_normals, reference_mesh
    )
    metrics = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "sampling": {
            "method": "deterministic_area_weighted_triangle_sampling",
            "distance_target": "exact_closest_point_on_triangle_mesh",
            "samples_per_mesh": args.samples,
            "seed": args.seed,
            "hausdorff_is_sampled_approximation": True,
            "trimesh_version": trimesh.__version__,
        },
        "reference_mesh": mesh_summary(reference_vertices, reference_triangles),
        "candidate_mesh": mesh_summary(candidate_vertices, candidate_triangles),
        "reference_to_candidate": reference_to_candidate,
        "candidate_to_reference": candidate_to_reference,
        "symmetric": {
            "chamfer_l1": 0.5
            * (
                reference_to_candidate["mean_distance"]
                + candidate_to_reference["mean_distance"]
            ),
            "chamfer_l2_squared": 0.5
            * (
                reference_to_candidate["mean_squared_distance"]
                + candidate_to_reference["mean_squared_distance"]
            ),
            "sampled_hausdorff": max(
                reference_to_candidate["max_distance"],
                candidate_to_reference["max_distance"],
            ),
            "mean_unoriented_normal_angle_degrees": 0.5
            * (
                reference_to_candidate["mean_unoriented_normal_angle_degrees"]
                + candidate_to_reference["mean_unoriented_normal_angle_degrees"]
            ),
        },
    }

    output = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

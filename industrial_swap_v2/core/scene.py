"""Lossless source Gaussian partitions and Graphdeco PLY composition."""
import numpy as np
from plyfile import PlyData, PlyElement
from .context import sha256, write_json


def read_vertices(path):
    return PlyData.read(str(path))["vertex"].data


def save_vertices(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(array, "vertex")], text=False).write(str(path))


def partition(source, target, keep, out):
    vertices = read_vertices(source)
    for indices in (target, keep):
        assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
    joined = np.concatenate((target, keep))
    assert len(joined) == len(vertices)
    assert np.array_equal(np.sort(joined), np.arange(len(vertices))), "Invalid source partition"
    save_vertices(out / "target_gaussians.ply", vertices[target])
    save_vertices(out / "background_gaussians.ply", vertices[keep])
    return len(vertices)


def compose(background_path, object_path, destination, record_path):
    background = read_vertices(background_path)
    obj = read_vertices(object_path)
    assert background.dtype.names == obj.dtype.names, "Graphdeco schema mismatch"
    combined = np.concatenate((background, obj.astype(background.dtype)))
    save_vertices(destination, combined)
    check = read_vertices(destination)
    assert check[:len(background)].tobytes() == background.tobytes()
    write_json(record_path, {"background_points": len(background), "new_object_points": len(obj),
               "total_points": len(check), "retained_background_bitwise_identical": True,
               "source_background_sha256": sha256(background_path), "object_sha256": sha256(object_path),
               "scene_sha256": sha256(destination)})

"""Check GSLE/COLMAP camera conversion against pinned DreamGaussian orbits."""

import sys, math
import numpy as np
from common import ROOT, look_at, save_json

sys.path.insert(0, str(ROOT / "repos/DreamGaussian"))
from cam_utils import orbit_camera

errors = []
for elevation in (-10, 0, 20, 60):
    for azimuth in (-150, -45, 0, 90, 179):
        pose = orbit_camera(-elevation, azimuth, radius=3.0)
        # OpenGL c2w -> COLMAP/OpenCV c2w (+y down,+z forward).
        pose[:3, 1:3] *= -1
        expected = np.linalg.inv(pose)
        e = math.radians(elevation)
        a = math.radians(azimuth)
        position = 3 * np.array(
            [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]
        )
        cam = look_at(position, np.zeros(3), np.array([0, 1, 0]), 512, 49.1)
        error = max(
            np.max(abs(cam.R - expected[:3, :3])), np.max(abs(cam.t - expected[:3, 3]))
        )
        assert error < 1e-6, (elevation, azimuth, error)
        errors.append(float(error))
record = {
    "tested_orbits": len(errors),
    "maximum_pose_error": max(errors),
    "elevation_conversion": "DreamGaussian = -GSLE elevation",
    "azimuth_conversion": "identical; positive moves from local+z toward local+x",
}
save_json(ROOT / "records/gsle/camera_validation.json", record)
print(record, flush=True)

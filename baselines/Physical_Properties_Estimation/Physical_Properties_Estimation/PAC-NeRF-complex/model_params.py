import json
import os

import numpy as np
import torch

from lib import constraint
from lib.engine import MPMSimulator


def tensor_to_python(value):
    if isinstance(value, torch.nn.Parameter):
        value = value.data
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def material_name(material):
    names = {
        MPMSimulator.elasticity: "elasticity",
        MPMSimulator.viscous_fluid: "viscous_fluid",
        MPMSimulator.von_mises: "von_mises",
        MPMSimulator.drucker_prager: "drucker_prager",
    }
    return names.get(material, str(material))


def collect_estimated_model_params(pnerf):
    params = {
        "object_initial_velocity": tensor_to_python(pnerf.global_v),
        "material": {
            "type": material_name(pnerf.material),
            "type_id": int(pnerf.material),
        },
        "material_parameters": {},
        "raw_optimized_parameters": {},
    }

    raw_names = [
        "global_v",
        "global_E",
        "global_nu",
        "global_mu",
        "global_lam",
        "global_kappa",
        "global_rho",
        "yield_stress",
        "plastic_viscosity",
        "plastic_pow",
        "friction_angle",
        "cohesion",
    ]
    for name in raw_names:
        if hasattr(pnerf, name):
            params["raw_optimized_parameters"][name] = tensor_to_python(getattr(pnerf, name))

    material_params = params["material_parameters"]
    if hasattr(pnerf, "global_E"):
        material_params["E"] = float((10 ** pnerf.global_E.detach()).cpu().item())
    if hasattr(pnerf, "global_nu"):
        material_params["nu"] = float(constraint(pnerf.global_nu.detach(), pnerf.nu_bound).cpu().item())
    if hasattr(pnerf, "global_mu"):
        material_params["mu"] = float((10 ** pnerf.global_mu.detach()).cpu().item())
    if hasattr(pnerf, "global_lam"):
        material_params["lambda"] = float((10 ** pnerf.global_lam.detach()).cpu().item())
    if hasattr(pnerf, "global_kappa"):
        material_params["kappa"] = float((10 ** pnerf.global_kappa.detach()).cpu().item())
    if hasattr(pnerf, "global_rho"):
        material_params["rho"] = float((10 ** pnerf.global_rho.detach()).cpu().item())
    if hasattr(pnerf, "yield_stress"):
        material_params["yield_stress"] = float((10 ** pnerf.yield_stress.detach()).cpu().item())
    if hasattr(pnerf, "plastic_viscosity"):
        material_params["plastic_viscosity"] = float((10 ** pnerf.plastic_viscosity.detach()).cpu().item())
    if hasattr(pnerf, "plastic_pow"):
        material_params["plastic_pow"] = tensor_to_python(pnerf.plastic_pow)
    if hasattr(pnerf, "friction_angle"):
        material_params["friction_angle"] = tensor_to_python(pnerf.friction_angle)
        sin_phi = torch.sin(pnerf.friction_angle.detach() / 180 * np.pi)
        friction_alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        material_params["friction_alpha"] = tensor_to_python(friction_alpha)
    if hasattr(pnerf, "cohesion"):
        material_params["cohesion"] = tensor_to_python(pnerf.cohesion)

    if hasattr(pnerf, "global_mu") and hasattr(pnerf, "global_kappa"):
        mu = 10 ** pnerf.global_mu.detach()
        kappa = 10 ** pnerf.global_kappa.detach()
        material_params["lambda_from_kappa_mu"] = float((kappa - 2.0 / 3.0 * mu).cpu().item())

    return params


def save_estimated_model_params(pnerf, cfg, args, stage="dynamic"):
    """Write physical (decoded) parameters in the dataset submission schema."""
    from property_export import property_output_dir

    output_dir = property_output_dir(cfg, args)
    if os.path.splitext(output_dir)[1].lower() == ".json":
        raise ValueError("--params-output must be a directory for the two JSON files")
    params = collect_estimated_model_params(pnerf)
    material_params = params["material_parameters"]
    if pnerf.material == MPMSimulator.elasticity:
        name, fields = "ELASTIC", {"E": "E", "nu": "nu"}
    elif pnerf.material == MPMSimulator.viscous_fluid:
        name, fields = "NEWTONIAN", {"mu": "mu", "kappa": "kappa"}
    elif pnerf.material == MPMSimulator.von_mises:
        if "E" in material_params:
            name, fields = "PLASTICINE", {"E": "E", "nu": "nu", "yield": "yield_stress"}
        else:
            name, fields = "NON_NEWTONIAN", {
                "mu": "mu", "kappa": "kappa", "yield": "yield_stress", "visc": "plastic_viscosity"}
    elif pnerf.material == MPMSimulator.drucker_prager:
        name, fields = "SAND", {"friction_angle": "friction_angle"}
    else:
        raise ValueError(f"Unsupported material: {pnerf.material}")
    material = {name: {output_key: material_params[source_key]
                       for output_key, source_key in fields.items()}}
    velocity = np.asarray(params["object_initial_velocity"], dtype=float).reshape(3)
    velocity = dict(zip(("vx", "vy", "vz"), velocity.tolist()))
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for filename, payload in (("materials_configs.json", material), ("initvelocity.json", velocity)):
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        paths.append(output_path)
        print(f"[Params] saved {output_path}")
    return tuple(paths)

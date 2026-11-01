"""Reject nonfinite checkpoint tensors before they are used."""
import torch


def assert_finite_tensors(value, context):
    invalid = []

    def visit(item, name):
        if torch.is_tensor(item):
            if not bool(torch.isfinite(item).all()):
                invalid.append(name)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{name}.{key}" if name else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{name}[{index}]")

    visit(value, "")
    if invalid:
        raise FloatingPointError(
            f"{context}: NaN/Inf in {', '.join(invalid)}. "
            "Stopped before continuing with invalid numerical state."
        )

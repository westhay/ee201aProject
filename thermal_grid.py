"""
Thermal Grid Module

This module creates a 3D voxel grid for thermal analysis.
It handles material assignment, effective properties for bonding layers,
and thermal resistance calculation.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def create_voxel_grid(boxes, voxel_size=0.1, layers=None, conductivity_values=None):
    """
    Create a 3D voxel grid from boxes and assign materials.

    Divides the system bounding box into uniform cubic voxels and assigns
    material, conductivity, power density, and box name to each voxel.
    Boxes are processed bottom-to-top so that higher boxes overwrite lower
    ones where they overlap.

    Args:
        boxes: List of Box objects with geometry and material info.
        voxel_size: Grid resolution in mm (default 0.1 mm = 100 microns).
        layers: List of Layer objects for material lookup (from
            layer_definitions.xml via parse_Layer_netlist).
        conductivity_values: Dict mapping material names to thermal
            conductivity in W/m·K.

    Returns:
        Dictionary with the following keys:
            'material_grid'    : np.ndarray (nx, ny, nz) dtype=object
                                 Material names per voxel.
            'conductivity_grid': np.ndarray (nx, ny, nz) dtype=float
                                 Thermal conductivity values (W/m·K).
            'power_grid'       : np.ndarray (nx, ny, nz) dtype=float
                                 Power density (W/m³).
            'box_grid'         : np.ndarray (nx, ny, nz) dtype=object
                                 Box names per voxel.
            'bounds'           : tuple (min_x, max_x, min_y, max_y,
                                        min_z, max_z) in mm.
            'voxel_size'       : float  Grid resolution in mm.
            'grid_shape'       : tuple  (nx, ny, nz).

    Raises:
        ValueError: If boxes is empty.
    """
    if not boxes:
        raise ValueError("boxes list cannot be empty")

    if conductivity_values is None:
        conductivity_values = {}

    # Step 1: Calculate system bounding box
    min_x = min(box.start_x for box in boxes)
    max_x = max(box.end_x for box in boxes)
    min_y = min(box.start_y for box in boxes)
    max_y = max(box.end_y for box in boxes)
    min_z = min(box.start_z for box in boxes)
    max_z = max(box.end_z for box in boxes)

    print(f"[Grid Creation] System bounds: X=[{min_x:.2f}, {max_x:.2f}] "
          f"Y=[{min_y:.2f}, {max_y:.2f}] Z=[{min_z:.2f}, {max_z:.2f}] mm")

    # Step 2: Calculate grid dimensions
    nx = int(np.ceil((max_x - min_x) / voxel_size))
    ny = int(np.ceil((max_y - min_y) / voxel_size))
    nz = int(np.ceil((max_z - min_z) / voxel_size))

    # Guard against degenerate dimensions
    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    total_voxels = nx * ny * nz
    print(f"[Grid Creation] Grid shape: ({nx}, {ny}, {nz}) = {total_voxels:,} voxels")
    print(f"[Grid Creation] Voxel size: {voxel_size} mm")

    # Step 3: Initialize grids
    air_k = conductivity_values.get('Air', 0.025)
    material_grid = np.full((nx, ny, nz), 'Air', dtype=object)
    conductivity_grid = np.full((nx, ny, nz), air_k, dtype=float)
    power_grid = np.zeros((nx, ny, nz), dtype=float)
    box_grid = np.full((nx, ny, nz), '', dtype=object)

    # Step 4: Fill voxels bottom-to-top so upper boxes overwrite lower ones
    sorted_boxes = sorted(boxes, key=lambda b: b.start_z)

    total_power = 0.0
    for box in sorted_boxes:
        # Determine material name and thermal conductivity for this box
        material, k_value = get_box_material(box, layers, conductivity_values)

        # Calculate voxel index ranges that overlap with this box
        i_start = max(0, int((box.start_x - min_x) / voxel_size))
        i_end = min(nx, int(np.ceil((box.end_x - min_x) / voxel_size)))
        j_start = max(0, int((box.start_y - min_y) / voxel_size))
        j_end = min(ny, int(np.ceil((box.end_y - min_y) / voxel_size)))
        k_start = max(0, int((box.start_z - min_z) / voxel_size))
        k_end = min(nz, int(np.ceil((box.end_z - min_z) / voxel_size)))

        # Assign material properties to the overlapping voxels
        material_grid[i_start:i_end, j_start:j_end, k_start:k_end] = material
        conductivity_grid[i_start:i_end, j_start:j_end, k_start:k_end] = k_value
        box_grid[i_start:i_end, j_start:j_end, k_start:k_end] = box.name

        # Distribute power density into voxels
        if box.power > 0:
            total_power += box.power
            num_voxels = (i_end - i_start) * (j_end - j_start) * (k_end - k_start)

            if num_voxels > 0:
                voxel_volume = (voxel_size * 1e-3) ** 3  # convert mm³ to m³

                if 'GPU' in box.name.upper():
                    # Distribute GPU power uniformly in the vertical center plane
                    z_center_idx = (k_start + k_end) // 2
                    num_center_voxels = (i_end - i_start) * (j_end - j_start)
                    if num_center_voxels > 0:
                        power_density = box.power / (num_center_voxels * voxel_volume)
                        power_grid[i_start:i_end, j_start:j_end, z_center_idx] = power_density
                else:
                    # Uniform distribution across all voxels of the box
                    power_density = box.power / (num_voxels * voxel_volume)
                    power_grid[i_start:i_end, j_start:j_end, k_start:k_end] = power_density

    # Step 5: Print summary statistics
    print(f"[Grid Creation] Total power: {total_power:.1f} W")
    print("[Grid Creation] Material distribution:")
    unique, counts = np.unique(material_grid, return_counts=True)
    for mat, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  - {mat}: {count:,} voxels ({100 * count / total_voxels:.1f}%)")

    return {
        'material_grid': material_grid,
        'conductivity_grid': conductivity_grid,
        'power_grid': power_grid,
        'box_grid': box_grid,
        'bounds': (min_x, max_x, min_y, max_y, min_z, max_z),
        'voxel_size': voxel_size,
        'grid_shape': (nx, ny, nz),
    }


def get_box_material(box, layers, conductivity_values):
    """
    Get material name and thermal conductivity from a Box.

    Handles three cases:

    1. **Bonding layer** – stackup has the form
       ``"1:Cu-Foil:70.0,Epoxy, Silver filled:30.0"`` where the ratios are
       percentages (0–100).  The effective conductivity is computed as a
       weighted average of the two constituent materials.

    2. **Layer stackup** – e.g. ``"1:5nm_active,9:5nm_global_metal"``.
       Each token is ``count:layer_name``.  The first layer whose name is
       found in *layers* and which has a defined material attribute is used.
       If that material is itself a composite (``"Cu-Foil:0.5,Si:0.5"``), an
       effective conductivity is computed.

    3. **Direct material name** – the stackup string is looked up directly
       in *conductivity_values*.

    Args:
        box: Box object with a ``stackup`` attribute (and optionally a
            ``get_box_stackup()`` method).
        layers: List of Layer objects from XML (may be None).
        conductivity_values: Dict mapping material names to thermal
            conductivity in W/m·K.

    Returns:
        tuple: (material_name: str, conductivity_value: float)
    """
    if conductivity_values is None:
        conductivity_values = {}

    stackup = (box.get_box_stackup()
               if hasattr(box, 'get_box_stackup')
               else box.stackup)

    if not stackup:
        return 'Air', conductivity_values.get('Air', 0.025)

    # ------------------------------------------------------------------
    # Case 1: Bonding layer with two-component format
    # Example: "1:Cu-Foil:70.0,Epoxy, Silver filled:30.0"
    # Detected by: first colon-group contains three colon-separated parts.
    # ------------------------------------------------------------------
    bonding_match = re.match(
        r'^(\d+):(.+?):([0-9.]+),(.+):([0-9.]+)$',
        stackup.strip()
    )
    if bonding_match:
        try:
            mat1_name = bonding_match.group(2).strip()
            ratio1 = float(bonding_match.group(3))
            mat2_name = bonding_match.group(4).strip()
            ratio2 = float(bonding_match.group(5))

            # Normalize ratios: therm.py stores them as percentages (0-100)
            if ratio1 > 1.0 or ratio2 > 1.0:
                total = ratio1 + ratio2
                if total > 0:
                    ratio1 /= total
                    ratio2 /= total

            # Resolve "Epoxy, Silver filled" alias to "EpAg"
            mat1_name = _resolve_material_alias(mat1_name, conductivity_values)
            mat2_name = _resolve_material_alias(mat2_name, conductivity_values)

            k1 = conductivity_values.get(mat1_name, 1.0)
            k2 = conductivity_values.get(mat2_name, 1.0)
            k_eff = ratio1 * k1 + ratio2 * k2

            effective_name = f"{mat1_name}_{mat2_name}_eff"
            return effective_name, k_eff
        except (ValueError, IndexError) as exc:
            print(f"[Warning] Could not parse bonding stackup '{stackup}': {exc}")

    # ------------------------------------------------------------------
    # Case 2: Layer stackup, e.g. "1:5nm_active,9:5nm_global_metal"
    # Each comma-separated token looks like "count:layer_name".
    # ------------------------------------------------------------------
    if layers and ':' in stackup:
        try:
            layer_names = parse_stackup_layers(stackup)
            for layer_name in layer_names:
                layer = find_layer_by_name(layers, layer_name)
                if layer is not None and hasattr(layer, 'material') and layer.material:
                    mat_str = layer.material.strip()
                    k_eff, mat_name = _parse_material_string(
                        mat_str, conductivity_values
                    )
                    return mat_name, k_eff
        except Exception as exc:
            print(f"[Warning] Could not parse layer stackup '{stackup}': {exc}")

    # ------------------------------------------------------------------
    # Case 3: Direct material name
    # ------------------------------------------------------------------
    material = stackup.strip()
    material = _resolve_material_alias(material, conductivity_values)
    k = conductivity_values.get(material, 1.0)
    return material, k


def calculate_voxel_resistances(grid_info):
    """
    Calculate thermal resistances for each voxel in X, Y, Z directions.

    Uses the formula R = L / (k * A).  For a cubic voxel of side *dx*:

    * R_x = dx / (k * dy * dz) = 1 / (k * dx)  (heat flow in X direction)
    * R_y = dy / (k * dx * dz) = 1 / (k * dy)  (heat flow in Y direction)
    * R_z = dz / (k * dx * dy) = 1 / (k * dz)  (heat flow in Z direction)

    Voxels with zero (or negative) conductivity are assigned a very high
    resistance of 1e6 K/W to represent air gaps or unknown materials.

    Args:
        grid_info: Dictionary returned by :func:`create_voxel_grid`.

    Returns:
        np.ndarray: Shape (nx, ny, nz, 3) where the last axis contains
        [R_x, R_y, R_z] in K/W for each voxel.
    """
    conductivity_grid = grid_info['conductivity_grid']
    voxel_size = grid_info['voxel_size']
    nx, ny, nz = grid_info['grid_shape']

    print("[Resistance Calculation] Computing thermal resistances...")

    # Convert mm to meters for SI units
    dx = dy = dz = voxel_size * 1e-3  # meters

    # Vectorized computation: R = L / (k * A).
    # For a cubic voxel: R_x = dx / (k * dy * dz) = 1 / (k * dx).
    # Voxels with k <= 0 (air/unknown) receive a high sentinel resistance.
    HIGH_R = 1e6  # K/W – representative of air or unknown material

    with np.errstate(divide='ignore', invalid='ignore'):
        r_uniform = np.where(
            conductivity_grid > 0,
            dx / (conductivity_grid * dy * dz),
            HIGH_R,
        )

    # Stack into (nx, ny, nz, 3) array.  All three directions are equal for
    # cubic voxels (dx == dy == dz), so all axes share the same array.
    resistance_grid = np.stack([r_uniform, r_uniform, r_uniform], axis=-1)

    # Print summary statistics for non-air voxels
    valid_mask = conductivity_grid > 0
    if np.any(valid_mask):
        R_x_avg = np.mean(resistance_grid[:, :, :, 0][valid_mask])
        R_y_avg = np.mean(resistance_grid[:, :, :, 1][valid_mask])
        R_z_avg = np.mean(resistance_grid[:, :, :, 2][valid_mask])

        print("[Resistance Calculation] Average resistances (K/W):")
        print(f"  R_x: {R_x_avg:.6e}")
        print(f"  R_y: {R_y_avg:.6e}")
        print(f"  R_z: {R_z_avg:.6e}")

    return resistance_grid


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_stackup_layers(stackup_string):
    """
    Parse a layer stackup string of the form ``"1:5nm_active,9:5nm_global_metal"``.

    Each comma-separated token is expected to be either:
    * ``"count:layer_name"`` – the layer name after the colon is returned, or
    * ``"layer_name"``       – returned as-is.

    Args:
        stackup_string: str  Stackup specification string.

    Returns:
        list[str]: Layer names (e.g. ``['5nm_active', '5nm_global_metal']``).
    """
    layer_names = []
    for part in stackup_string.split(','):
        part = part.strip()
        if ':' in part:
            # Format "count:layer_name" – take everything after the first colon
            layer_name = part.split(':', 1)[1]
            layer_names.append(layer_name)
        else:
            layer_names.append(part)
    return layer_names


def find_layer_by_name(layers, layer_name):
    """
    Find a Layer object by its name attribute.

    Args:
        layers: List of Layer objects (each must have a ``name`` attribute).
        layer_name: str  Name to search for.

    Returns:
        Layer object if found, otherwise ``None``.
    """
    if not layers:
        return None
    for layer in layers:
        if hasattr(layer, 'name') and layer.name == layer_name:
            return layer
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_material_alias(material_name, conductivity_values):
    """
    Translate known material aliases to the canonical name used in
    *conductivity_values*.

    For example, ``"Epoxy, Silver filled"`` is stored as ``"EpAg"`` in the
    conductivity dict.  If the raw name is already present in the dict, it
    is returned unchanged.

    Args:
        material_name: str  Raw material name from the stackup or XML.
        conductivity_values: Dict mapping canonical names to conductivity.

    Returns:
        str: Resolved material name.
    """
    # Already a known material – no translation needed
    if material_name in conductivity_values:
        return material_name

    # Known aliases
    _aliases = {
        'Epoxy, Silver filled': 'EpAg',
        'epoxy, silver filled': 'EpAg',
        'Epoxy Silver filled': 'EpAg',
    }
    return _aliases.get(material_name, material_name)


def _parse_material_string(material_str, conductivity_values):
    """
    Parse a material string that may be a single material or a composite.

    Composite format (as seen in layer_definitions.xml):
    ``"Cu-Foil:0.5,Si:0.5"`` where ratios are in the range [0, 1].

    Args:
        material_str: str  Raw material string from a Layer object.
        conductivity_values: Dict mapping material names to conductivity.

    Returns:
        tuple: (effective_conductivity: float, material_name: str)
    """
    if ':' not in material_str:
        # Single material
        mat = _resolve_material_alias(material_str.strip(), conductivity_values)
        return conductivity_values.get(mat, 1.0), mat

    # Composite material: "Mat1:ratio1,Mat2:ratio2"
    try:
        parts = material_str.split(',')
        if len(parts) == 2:
            mat1_parts = parts[0].rsplit(':', 1)
            mat2_parts = parts[1].rsplit(':', 1)
            if len(mat1_parts) == 2 and len(mat2_parts) == 2:
                mat1 = _resolve_material_alias(
                    mat1_parts[0].strip(), conductivity_values
                )
                r1 = float(mat1_parts[1])
                mat2 = _resolve_material_alias(
                    mat2_parts[0].strip(), conductivity_values
                )
                r2 = float(mat2_parts[1])

                k1 = conductivity_values.get(mat1, 1.0)
                k2 = conductivity_values.get(mat2, 1.0)
                k_eff = r1 * k1 + r2 * k2
                return k_eff, f"{mat1}_{mat2}_eff"
    except (ValueError, IndexError):
        pass

    # Fallback: treat the whole string as a material name
    mat = _resolve_material_alias(material_str.strip(), conductivity_values)
    return conductivity_values.get(mat, 1.0), mat


# ---------------------------------------------------------------------------
# Minimal self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rearrange import Box

    print("=== thermal_grid.py self-test ===\n")

    conductivity_values = {
        'Air': 0.025,
        'Si': 105.0,
        'Cu-Foil': 400.0,
        'EpAg': 1.6,
        'TIM': 100.0,
        'FR-4': 0.1,
    }

    # Create a minimal set of test boxes
    # Interposer: 30 mm x 30 mm x 0.5 mm, Si
    interposer = Box(0.0, 0.0, 0.0, 30.0, 30.0, 0.5, 0.0, 'Si', 0.0, 'interposer')

    # GPU: 26 mm x 32 mm x 0.7 mm, 270 W
    gpu = Box(2.0, 0.0, 0.5, 26.0, 32.0, 0.7, 270.0, 'Si', 0.0, 'GPU#0')

    # One HBM: 7 mm x 11 mm x 0.7 mm, 5 W
    hbm = Box(0.0, 0.0, 0.5, 7.0, 11.0, 0.7, 5.0, 'Si', 0.0, 'HBM#0')

    boxes = [interposer, gpu, hbm]

    print("Creating voxel grid with 1.0 mm resolution (coarse for speed)...")
    grid_info = create_voxel_grid(
        boxes,
        voxel_size=1.0,
        conductivity_values=conductivity_values,
    )

    print(f"\nGrid shape : {grid_info['grid_shape']}")
    print(f"Bounds     : {grid_info['bounds']}")

    print("\nCalculating thermal resistances...")
    resistance_grid = calculate_voxel_resistances(grid_info)
    print(f"Resistance grid shape: {resistance_grid.shape}")

    print("\nGrid creation and resistance calculation successful!")

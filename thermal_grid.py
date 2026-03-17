"""
Thermal Grid Module

This module creates a 3D voxel grid for thermal analysis.
It handles material assignment, effective properties for bonding layers,
and thermal resistance calculation.
"""

import csv
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PySpice.Spice.Netlist import Circuit

def export_boxes_to_csv(box_list, filename):
    """
    Export a list of Box objects to a CSV file.
    """

    # Define CSV column headers
    headers = [
        "name",
        "start_x", "start_y", "start_z",
        "width", "length", "height",
        "end_x", "end_y", "end_z",
        "power",
        "stackup"
    ]

    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(headers)

        # Write each box's attributes
        for box in box_list:
            # Safely compute end coordinates if not explicitly stored
            end_x = box.start_x + box.width
            end_y = box.start_y + box.length
            end_z = box.start_z + box.height

            writer.writerow([
                box.name,
                box.start_x, box.start_y, box.start_z,
                box.width, box.length, box.height,
                end_x, end_y, end_z,
                box.power,
                box.stackup
            ])

def create_voxel_grid(boxes, voxel_size=0.1, layers=None, conductivity_values=None):
    """
    Create a 3D voxel grid from boxes and assign materials.

    Divides the system bounding box into uniform cubic voxels and assigns
    material, conductivity, power density, and box name to each voxel.
    Boxes are processed bottom-to-top so that higher boxes overwrite lower
    ones where they overlap.

    Power is assigned based on geometric overlap with the box bounds,
    NOT based on final box ownership in box_grid. This is important because
    power-source boxes may be overwritten by later structural/material boxes
    but should still inject heat into the overlapped voxels.

    Args:
        boxes: List of Box objects with geometry and material info.
        voxel_size: Grid resolution in mm.
        layers: List of Layer objects for material lookup.
        conductivity_values: Dict mapping material names to thermal conductivity.

    Returns:
        Dictionary with:
            'material_grid'
            'conductivity_grid'
            'power_grid'
            'box_grid'
            'bounds'
            'voxel_size'
            'grid_shape'
            'active_mask'
    """
    if not boxes:
        raise ValueError("boxes list cannot be empty")

    if conductivity_values is None:
        conductivity_values = {}

    # ---------------------------------------------------------
    # Step 1: Calculate system bounding box
    # ---------------------------------------------------------
    min_x = min(box.start_x for box in boxes)
    max_x = max(box.end_x for box in boxes)
    min_y = min(box.start_y for box in boxes)
    max_y = max(box.end_y for box in boxes)
    min_z = min(box.start_z for box in boxes)
    max_z = max(box.end_z for box in boxes)

    print(f"[Grid Creation] System bounds: X=[{min_x:.2f}, {max_x:.2f}] "
          f"Y=[{min_y:.2f}, {max_y:.2f}] Z=[{min_z:.2f}, {max_z:.2f}] mm")

    # ---------------------------------------------------------
    # Step 2: Calculate grid dimensions
    # ---------------------------------------------------------
    nx = int(np.ceil((max_x - min_x) / voxel_size))
    ny = int(np.ceil((max_y - min_y) / voxel_size))
    nz = int(np.ceil((max_z - min_z) / voxel_size))

    nx = max(nx, 1)
    ny = max(ny, 1)
    nz = max(nz, 1)

    total_voxels = nx * ny * nz
    print(f"[Grid Creation] Grid shape: ({nx}, {ny}, {nz}) = {total_voxels:,} voxels")
    print(f"[Grid Creation] Voxel size: {voxel_size} mm")

    # ---------------------------------------------------------
    # Step 3: Initialize grids
    # ---------------------------------------------------------
    air_k = conductivity_values.get('Air', 0.025)
    material_grid = np.full((nx, ny, nz), 'Air', dtype=object)
    conductivity_grid = np.full((nx, ny, nz), air_k, dtype=float)
    power_grid = np.zeros((nx, ny, nz), dtype=float)
    box_grid = np.full((nx, ny, nz), '', dtype=object)

    # ---------------------------------------------------------
    # Step 4: Assign materials/conductivities first
    # ---------------------------------------------------------
    sorted_boxes = sorted(boxes, key=lambda b: b.start_z)

    for box in sorted_boxes:
        material, k_value = get_box_material(box, layers, conductivity_values)

        i_start = max(0, int((box.start_x - min_x) / voxel_size))
        i_end = min(nx, int(np.ceil((box.end_x - min_x) / voxel_size)))
        j_start = max(0, int((box.start_y - min_y) / voxel_size))
        j_end = min(ny, int(np.ceil((box.end_y - min_y) / voxel_size)))
        k_start = max(0, int((box.start_z - min_z) / voxel_size))
        k_end = min(nz, int(np.ceil((box.end_z - min_z) / voxel_size)))

        material_grid[i_start:i_end, j_start:j_end, k_start:k_end] = material
        conductivity_grid[i_start:i_end, j_start:j_end, k_start:k_end] = k_value
        box_grid[i_start:i_end, j_start:j_end, k_start:k_end] = box.name

    # ---------------------------------------------------------
    # Step 5: Assign power using GEOMETRIC OVERLAP
    # ---------------------------------------------------------
    voxel_volume = (voxel_size * 1e-3) ** 3  # mm^3 -> m^3
    expected_total_power = 0.0

    for box in sorted_boxes:
        box_power = getattr(box, 'power', 0.0)
        if box_power <= 0:
            continue

        expected_total_power += box_power

        i_start = max(0, int((box.start_x - min_x) / voxel_size))
        i_end = min(nx, int(np.ceil((box.end_x - min_x) / voxel_size)))
        j_start = max(0, int((box.start_y - min_y) / voxel_size))
        j_end = min(ny, int(np.ceil((box.end_y - min_y) / voxel_size)))
        k_start = max(0, int((box.start_z - min_z) / voxel_size))
        k_end = min(nz, int(np.ceil((box.end_z - min_z) / voxel_size)))

        assigned_voxels = []

        # Only the true GPU die uses center-plane power injection
        is_gpu = box.name.endswith(".GPU")

        if is_gpu:
            if k_end > k_start:
                z_center_idx = (k_start + k_end) // 2
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        assigned_voxels.append((i, j, z_center_idx))
        else:
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        assigned_voxels.append((i, j, k))

        num_assigned = len(assigned_voxels)

        if num_assigned == 0:
            print(f"[Warning] No assigned voxels found for powered box '{box.name}'")
            continue

        power_density = box_power / (num_assigned * voxel_volume)

        for (i, j, k) in assigned_voxels:
            power_grid[i, j, k] += power_density

    # ---------------------------------------------------------
    # Step 6: Print summary statistics
    # ---------------------------------------------------------
    print(f"[Grid Creation] Expected total box power: {expected_total_power:.6f} W")

    total_power_from_grid = power_grid.sum() * voxel_volume
    print(f"[Grid Creation] Total power from grid:   {total_power_from_grid:.6f} W")
    print(f"[Grid Creation] Power error:             "
          f"{total_power_from_grid - expected_total_power:.6f} W")

    print("[Grid Creation] Material distribution:")
    unique, counts = np.unique(material_grid, return_counts=True)
    for mat, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  - {mat}: {count:,} voxels ({100 * count / total_voxels:.1f}%)")

    active_mask = box_grid != ''

    return {
        'material_grid': material_grid,
        'conductivity_grid': conductivity_grid,
        'power_grid': power_grid,
        'box_grid': box_grid,
        'bounds': (min_x, max_x, min_y, max_y, min_z, max_z),
        'voxel_size': voxel_size,
        'grid_shape': (nx, ny, nz),
        'active_mask': active_mask
    }

def get_box_material(box, layers, conductivity_values):
    """
    Get material name and thermal conductivity from a Box.

    Handles three cases:

    1. Bonding layer:
       Example: "1:Cu-Foil:70.0,Epoxy, Silver filled:30.0"
       Effective conductivity is computed as a weighted average of the two
       constituent materials.

    2. Layer stackup:
       Example: "1:5nm_active,9:5nm_global_metal"
       Each token is "count:layer_name". We compute an effective conductivity
       using a thickness-aware SERIES model:
           k_eff = total_thickness / sum(thickness_i / k_i)

       where thickness_i includes the token count multiplier.

    2.5 Single count-prefixed layer/material:
       Example: "1:TIM0p5" or "1:dummySi_HBM"
       First try to resolve the RHS as a layer name from XML. If found, use that
       layer's material. Otherwise treat it as a direct material key.

    3. Direct material name:
       Example: "Si"
       The stackup string is looked up directly in conductivity_values.
       If not found, and it matches a layer name, use that layer's material.
    """
    if conductivity_values is None:
        conductivity_values = {}

    stackup = (box.get_box_stackup()
               if hasattr(box, 'get_box_stackup')
               else box.stackup)

    if not stackup:
        return 'Air', conductivity_values.get('Air', 0.025)

    stackup = stackup.strip()

    # ------------------------------------------------------------------
    # Case 1: Bonding layer with two-component format
    # Example: "1:Cu-Foil:70.0,Epoxy, Silver filled:30.0"
    # ------------------------------------------------------------------
    bonding_match = re.match(
        r'^(\d+):(.+?):([0-9.]+),(.+):([0-9.]+)$',
        stackup
    )
    if bonding_match:
        try:
            mat1_name = bonding_match.group(2).strip()
            ratio1 = float(bonding_match.group(3))
            mat2_name = bonding_match.group(4).strip()
            ratio2 = float(bonding_match.group(5))

            # Normalize percentages if needed
            if ratio1 > 1.0 or ratio2 > 1.0:
                total = ratio1 + ratio2
                if total > 0:
                    ratio1 /= total
                    ratio2 /= total

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
    # Compute effective conductivity using thickness-aware SERIES model.
    # Only return from Case 2 if at least one token resolves to a real layer.
    # ------------------------------------------------------------------
    if layers and ',' in stackup:
        try:
            total_thickness = 0.0
            total_r_per_area = 0.0   # sum(thickness / k)
            resolved_material_names = []

            for token in stackup.split(','):
                token = token.strip()
                if not token:
                    continue

                count = 1.0
                layer_name = token

                if ':' in token:
                    left, right = token.split(':', 1)
                    layer_name = right.strip()
                    try:
                        count = float(left.strip())
                    except ValueError:
                        count = 1.0

                layer = find_layer_by_name(layers, layer_name)
                if layer is None or not hasattr(layer, 'material') or not layer.material:
                    continue

                mat_str = layer.material.strip()
                k_layer, mat_name = _parse_material_string(mat_str, conductivity_values)

                # Thickness comes from XML; include token count multiplier
                try:
                    layer_thickness = float(getattr(layer, 'thickness', 0.0))
                except (TypeError, ValueError):
                    layer_thickness = 0.0

                eff_thickness = count * layer_thickness

                # Skip degenerate/non-physical entries safely
                if eff_thickness <= 0:
                    continue
                if k_layer <= 0:
                    k_layer = 1e-12

                total_thickness += eff_thickness
                total_r_per_area += eff_thickness / k_layer
                resolved_material_names.append(mat_name)

            if total_thickness > 0 and total_r_per_area > 0:
                k_eff = total_thickness / total_r_per_area

                unique_mats = []
                for name in resolved_material_names:
                    if name not in unique_mats:
                        unique_mats.append(name)

                if len(unique_mats) == 1:
                    effective_name = unique_mats[0]
                else:
                    effective_name = "stackup_eff[" + "+".join(unique_mats[:3])
                    if len(unique_mats) > 3:
                        effective_name += "+..."
                    effective_name += "]"

                return effective_name, k_eff

        except Exception as exc:
            print(f"[Warning] Could not parse layer stackup '{stackup}': {exc}")

    # ------------------------------------------------------------------
    # Case 2.5: Single count-prefixed layer/material, e.g. "1:TIM0p5"
    # First try resolving the RHS as a layer name from XML.
    # ------------------------------------------------------------------
    if ':' in stackup and ',' not in stackup:
        parts = stackup.split(':', 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            rhs_name = parts[1].strip()

            # First: try XML layer resolution
            if layers:
                layer = find_layer_by_name(layers, rhs_name)
                if layer is not None and hasattr(layer, 'material') and layer.material:
                    mat_str = layer.material.strip()
                    k_layer, mat_name = _parse_material_string(mat_str, conductivity_values)
                    return mat_name, k_layer

            # Second: treat RHS as direct material key
            material = _resolve_material_alias(rhs_name, conductivity_values)
            k = conductivity_values.get(material, 1.0)
            return material, k

    # ------------------------------------------------------------------
    # Case 3: Direct material name.
    # If not found directly, also try resolving it as an XML layer name.
    # This fixes strings like "dummySi_HBM" if they appear without count prefix.
    # ------------------------------------------------------------------
    material = _resolve_material_alias(stackup, conductivity_values)
    if material in conductivity_values:
        return material, conductivity_values.get(material, 1.0)

    if layers:
        layer = find_layer_by_name(layers, stackup)
        if layer is not None and hasattr(layer, 'material') and layer.material:
            mat_str = layer.material.strip()
            k_layer, mat_name = _parse_material_string(mat_str, conductivity_values)
            return mat_name, k_layer

    return material, conductivity_values.get(material, 1.0)

def calculate_voxel_resistances(grid_info):
    """
    Calculate thermal resistances for each voxel in X, Y, Z directions.

    Uses the voxel formula R = L / (k * A) to match the
    interface_resistance() used in the circuit builder.  For a cubic voxel
    of side *dx*:

    * R_x = dx / (k * dy * dz)  
    * R_y = dy / (k * dx * dz)
    * R_z = dz / (k * dx * dy)

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

    # Vectorized computation: R = (L/2) / (k * A).
    # For a cubic voxel (dx == dy == dz): R_x = (dx/2) / (k * dy * dz).
    # Voxels with k <= 0 (air/unknown) receive a high sentinel resistance.
    HIGH_R = 1e6  # K/W – representative of air or unknown material

    with np.errstate(divide='ignore', invalid='ignore'):
        r_uniform = np.where(
            conductivity_grid > 0,
            dx / (conductivity_grid * dy * dz),  # half-voxel: matches interface_resistance()
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


# ---------------------------------------------------------
# Helper: unique node name for each voxel
# ---------------------------------------------------------
def voxel_node(i, j, k):
    return f"n_{i}_{j}_{k}"


# ---------------------------------------------------------
# Helper: thermal resistance between two neighboring voxels
# direction is one of 'x', 'y', 'z'
#
# Uses center-to-center resistance:
#   R = (L1 / (k1*A)) + (L2 / (k2*A))
#
# For equal cubic voxels:
#   L1 = L2 = d/2
# ---------------------------------------------------------
def interface_resistance(k1, k2, d_m, direction):
    if k1 <= 0:
        k1 = 1e-12
    if k2 <= 0:
        k2 = 1e-12

    if direction == 'x':
        area = d_m * d_m   # dy * dz
    elif direction == 'y':
        area = d_m * d_m   # dx * dz
    elif direction == 'z':
        area = d_m * d_m   # dx * dy
    else:
        raise ValueError("direction must be 'x', 'y', or 'z'")

    return (d_m / 2.0) / (k1 * area) + (d_m / 2.0) / (k2 * area)


# ---------------------------------------------------------
# Optional helper: convection / ambient resistance
# R = 1 / (h*A)
# h in W/(m^2*K)
# ---------------------------------------------------------
def boundary_resistance(h, area_m2):
    if h is None or h <= 0:
        return None
    return 1.0 / (h * area_m2)


# ---------------------------------------------------------
# Build PySpice circuit from voxel grid
# ---------------------------------------------------------
def build_thermal_circuit_from_grid(
    conductivity_grid,
    power_grid,
    voxel_size_mm,
    h_top=1000.0,
    h_side=10.0,
    h_bottom=100.0,
    active_mask=None
):
    """
    Parameters
    ----------
    conductivity_grid : np.ndarray, shape (nx, ny, nz)
        Thermal conductivity in W/(m*K)
    power_grid : np.ndarray, shape (nx, ny, nz)
        Power density in W/m^3
    voxel_size_mm : float
        Cubic voxel edge length in mm
    h_top, h_side, h_bottom : float or None
        Ambient boundary coefficients in W/(m^2*K)
    active_mask : np.ndarray of bool, optional
        If provided, only active voxels are connected/simulated.
        If None, all voxels are included.

    Returns
    -------
    circuit : PySpice Circuit
    """
    active_voxel_count = 0

    rx_count = 0
    ry_count = 0
    rz_count = 0
    
    top_bc_count = 0
    bottom_bc_count = 0
    side_bc_count = 0
    total_exposed_faces = 0
    
    nx, ny, nz = conductivity_grid.shape

    if active_mask is None:
        active_mask = np.ones((nx, ny, nz), dtype=bool)

    circuit = Circuit("3D Thermal Network")

    # convert mm -> m
    d_m = voxel_size_mm * 1e-3
    voxel_volume_m3 = d_m ** 3
    face_area_m2 = d_m ** 2

    resistor_count = 0
    source_count = 0
    ambient_count = 0

    # --------------------------------------------------
    # 1) Neighbor resistors
    # Only connect in +x, +y, +z to avoid duplicates
    # --------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not active_mask[i, j, k]:
                    continue

                active_voxel_count += 1
                n1 = voxel_node(i, j, k)
                k1 = conductivity_grid[i, j, k]

                # +x neighbor
                if i + 1 < nx and active_mask[i + 1, j, k]:
                    k2 = conductivity_grid[i + 1, j, k]
                    R = interface_resistance(k1, k2, d_m, 'x')
                    circuit.R(f"rx_{resistor_count}", n1, voxel_node(i + 1, j, k), R)
                    resistor_count += 1
                    rx_count += 1

                # +y neighbor
                if j + 1 < ny and active_mask[i, j + 1, k]:
                    k2 = conductivity_grid[i, j + 1, k]
                    R = interface_resistance(k1, k2, d_m, 'y')
                    circuit.R(f"ry_{resistor_count}", n1, voxel_node(i, j + 1, k), R)
                    resistor_count += 1
                    ry_count += 1

                # +z neighbor
                if k + 1 < nz and active_mask[i, j, k + 1]:
                    k2 = conductivity_grid[i, j, k + 1]
                    R = interface_resistance(k1, k2, d_m, 'z')
                    circuit.R(f"rz_{resistor_count}", n1, voxel_node(i, j, k + 1), R)
                    resistor_count += 1
                    rz_count += 1

    # --------------------------------------------------
    # 2) Heat injection current sources
    #
    # power_grid is in W/m^3, so voxel power:
    #   P_voxel = power_density * voxel_volume
    #
    # In thermal-electric analogy:
    #   heat flow -> current
    # --------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not active_mask[i, j, k]:
                    continue

                p_density = power_grid[i, j, k]
                p_voxel = p_density * voxel_volume_m3

                if abs(p_voxel) > 0:
                    circuit.I(f"p_{source_count}", circuit.gnd, voxel_node(i, j, k), p_voxel)
                    source_count += 1

    # --------------------------------------------------
    # 3) Ambient boundary resistors
    #
    # For each exposed face, connect to ground through R = 1/(hA)
    # --------------------------------------------------
    R_top = boundary_resistance(h_top, face_area_m2)
    R_side = boundary_resistance(h_side, face_area_m2)
    R_bottom = boundary_resistance(h_bottom, face_area_m2)

    def is_exposed(i2, j2, k2):
        return (
            i2 < 0 or i2 >= nx or
            j2 < 0 or j2 >= ny or
            k2 < 0 or k2 >= nz
        )

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not active_mask[i, j, k]:
                    continue

                n = voxel_node(i, j, k)

                # top (+z)
                if is_exposed(i, j, k + 1) and R_top is not None:
                    circuit.R(f"rtop_{ambient_count}", n, circuit.gnd, R_top)
                    ambient_count += 1
                    top_bc_count += 1
                    total_exposed_faces += 1

                # bottom (-z)
                if is_exposed(i, j, k - 1) and R_bottom is not None:
                    circuit.R(f"rbot_{ambient_count}", n, circuit.gnd, R_bottom)
                    ambient_count += 1
                    bottom_bc_count += 1
                    total_exposed_faces += 1

                # x sides
                if is_exposed(i - 1, j, k) and R_side is not None:
                    circuit.R(f"rxm_{ambient_count}", n, circuit.gnd, R_side)
                    ambient_count += 1
                    side_bc_count += 1
                    total_exposed_faces += 1
                if is_exposed(i + 1, j, k) and R_side is not None:
                    circuit.R(f"rxp_{ambient_count}", n, circuit.gnd, R_side)
                    ambient_count += 1
                    side_bc_count += 1
                    total_exposed_faces += 1

                # y sides
                if is_exposed(i, j - 1, k) and R_side is not None:
                    circuit.R(f"rym_{ambient_count}", n, circuit.gnd, R_side)
                    ambient_count += 1
                    side_bc_count += 1
                    total_exposed_faces += 1
                if is_exposed(i, j + 1, k) and R_side is not None:
                    circuit.R(f"ryp_{ambient_count}", n, circuit.gnd, R_side)
                    ambient_count += 1
                    side_bc_count += 1
                    total_exposed_faces += 1

    return circuit


# ---------------------------------------------------------
# Solve temperature map
# ---------------------------------------------------------
def solve_temperature_grid(
    conductivity_grid,
    power_grid,
    voxel_size_mm,
    T_ambient=25.0,
    h_top=1000.0,
    h_side=10.0,
    h_bottom=100.0,
    active_mask=None
):
    """
    Returns absolute temperature grid in degC.
    """
    circuit = build_thermal_circuit_from_grid(
        conductivity_grid=conductivity_grid,
        power_grid=power_grid,
        voxel_size_mm=voxel_size_mm,
        h_top=h_top,
        h_side=h_side,
        h_bottom=h_bottom,
        active_mask=active_mask
    )

    simulator = circuit.simulator(
        temperature=T_ambient,
        nominal_temperature=T_ambient
    )
    analysis = simulator.operating_point()

    nx, ny, nz = conductivity_grid.shape
    temperature_grid = np.full((nx, ny, nz), T_ambient, dtype=float)

    if active_mask is None:
        active_mask = np.ones((nx, ny, nz), dtype=bool)

    active_indices = np.argwhere(active_mask)
    for idx in active_indices:
        i, j, k = idx
        node_name = voxel_node(i, j, k)
        try:
            temp_rise = float(analysis[node_name])
        except Exception:
            temp_rise = 0.0
        temperature_grid[i, j, k] = T_ambient + temp_rise

    return temperature_grid, circuit, analysis

def summarize_temperature_grid(temperature_grid, active_mask=None, voxel_size_mm=None, bounds=None, T_ambient=25.0):
    import numpy as np

    if active_mask is None:
        active_mask = np.ones_like(temperature_grid, dtype=bool)

    active_temps = temperature_grid[active_mask]

    max_temp = np.max(active_temps)
    min_temp = np.min(active_temps)
    avg_temp = np.mean(active_temps)

    masked_grid = np.where(active_mask, temperature_grid, -np.inf)
    max_idx_flat = np.argmax(masked_grid)
    hottest_idx = np.unravel_index(max_idx_flat, temperature_grid.shape)

    summary = {
        "ambient_temp_C": float(T_ambient),
        "grid_shape": tuple(int(x) for x in temperature_grid.shape),
        "active_voxel_count": int(np.sum(active_mask)),
        "max_temp_C": float(max_temp),
        "min_temp_C": float(min_temp),
        "avg_temp_C": float(avg_temp),
        "hottest_voxel_index": tuple(int(x) for x in hottest_idx),
    }

    if voxel_size_mm is not None and bounds is not None:
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        i, j, k = hottest_idx
        x_mm = min_x + (i + 0.5) * voxel_size_mm
        y_mm = min_y + (j + 0.5) * voxel_size_mm
        z_mm = min_z + (k + 0.5) * voxel_size_mm

        summary["hottest_voxel_center_mm"] = (
            float(x_mm), float(y_mm), float(z_mm)
        )
    return summary

def write_temperature_report(summary, output_path="temperature_summary.txt"):
    with open(output_path, "w") as f:
        f.write("Thermal Simulation Summary\n")
        f.write("==========================\n\n")

        f.write(f"Ambient Temperature (C): {summary['ambient_temp_C']:.3f}\n")
        f.write(f"Grid Shape: {summary['grid_shape']}\n")
        f.write(f"Active Voxel Count: {summary['active_voxel_count']}\n\n")

        f.write(f"Maximum Temperature (C): {summary['max_temp_C']:.6f}\n")
        f.write(f"Minimum Temperature (C): {summary['min_temp_C']:.6f}\n")
        f.write(f"Average Temperature (C): {summary['avg_temp_C']:.6f}\n")
        f.write(f"Hottest Voxel Index: {summary['hottest_voxel_index']}\n")

        if 'hottest_voxel_center_mm' in summary:
            f.write(
                f"Hottest Voxel Center (mm): {summary['hottest_voxel_center_mm']}\n"
            )

    print(f"✓ Created {output_path}")
def summarize_temperatures_by_box(
    temperature_grid: np.ndarray,
    box_grid: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Returns dict: {box_name: (peak_temp_C, avg_temp_C)}
    """
    if active_mask is None:
        active_mask = np.ones_like(temperature_grid, dtype=bool)

    # Only consider voxels that belong to a named box and are active
    valid = active_mask & (box_grid != "")

    results: Dict[str, Tuple[float, float]] = {}
    box_names = np.unique(box_grid[valid])

    for name in box_names:
        m = valid & (box_grid == name)
        temps = temperature_grid[m]
        if temps.size == 0:
            continue
        results[str(name)] = (float(np.max(temps)), float(np.mean(temps)))

    return results


def summarize_resistances_by_box(
    resistance_grid: np.ndarray,   # shape (nx,ny,nz,3)
    box_grid: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    reducer: str = "mean",         # "mean" or "median"
) -> Dict[str, Tuple[float, float, float]]:
    """
    Returns dict: {box_name: (R_x, R_y, R_z)} in K/W.

    NOTE: This is a simple reduction across voxels in the box.
    """
    if active_mask is None:
        active_mask = np.ones(box_grid.shape, dtype=bool)

    valid = active_mask & (box_grid != "")

    if reducer not in ("mean", "median"):
        raise ValueError("reducer must be 'mean' or 'median'")

    fn = np.mean if reducer == "mean" else np.median

    results: Dict[str, Tuple[float, float, float]] = {}
    box_names = np.unique(box_grid[valid])

    for name in box_names:
        m = valid & (box_grid == name)
        r = resistance_grid[m]  # shape (N,3)
        if r.size == 0:
            continue
        rx, ry, rz = fn(r[:, 0]), fn(r[:, 1]), fn(r[:, 2])
        results[str(name)] = (float(rx), float(ry), float(rz))

    return results


def summarize_by_box_list(temperature_grid, resistance_grid, all_boxes, grid_info):
    """
    Instead of using box_grid string matching, iterate over the original
    all_boxes list and compute voxel index ranges directly from geometry.
    This guarantees every box in all_boxes gets a row in the output,
    with the correct .name used as the key.
    """
    voxel_size = grid_info['voxel_size']
    min_x, max_x, min_y, max_y, min_z, max_z = grid_info['bounds']
    nx, ny, nz = grid_info['grid_shape']
    active_mask = grid_info.get('active_mask', None)

    results = {}
    for box in all_boxes:
        i_start = max(0, int((box.start_x - min_x) / voxel_size))
        i_end   = min(nx, int(np.ceil((box.end_x   - min_x) / voxel_size)))
        j_start = max(0, int((box.start_y - min_y) / voxel_size))
        j_end   = min(ny, int(np.ceil((box.end_y   - min_y) / voxel_size)))
        k_start = max(0, int((box.start_z - min_z) / voxel_size))
        k_end   = min(nz, int(np.ceil((box.end_z   - min_z) / voxel_size)))

        if i_end <= i_start or j_end <= j_start or k_end <= k_start:
            continue  # degenerate box, skip

        t_slice = temperature_grid[i_start:i_end, j_start:j_end, k_start:k_end]
        r_slice = resistance_grid[i_start:i_end, j_start:j_end, k_start:k_end]  # shape (…,3)

        if active_mask is not None:
            mask_slice = active_mask[i_start:i_end, j_start:j_end, k_start:k_end]
            t_vals = t_slice[mask_slice]
            r_vals = r_slice[mask_slice]  # shape (N, 3)
        else:
            t_vals = t_slice.ravel()
            r_vals = r_slice.reshape(-1, 3)

        if t_vals.size == 0:
            continue

        t_peak = float(np.max(t_vals))
        t_avg  = float(np.mean(t_vals))
        rx = float(np.mean(r_vals[:, 0]))
        ry = float(np.mean(r_vals[:, 1]))
        rz = float(np.mean(r_vals[:, 2]))

        results[box.name] = (t_peak, t_avg, rx, ry, rz)

    return results

def write_box_results_report(results: dict, output_path="box_results.txt"):
    with open(output_path, "w") as f:
        f.write("Box Results (peak_temp_C, avg_temp_C, R_x, R_y, R_z)\n")
        f.write("====================================================\n")
        for name in sorted(results.keys()):
            peak, avg, rx, ry, rz = results[name]
            f.write(f"{name}, {peak:.6f}, {avg:.6f}, {rx:.6e}, {ry:.6e}, {rz:.6e}\n")


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

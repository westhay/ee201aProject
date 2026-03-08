"""
Thermal Simulator - simulator_simulate() function

This module implements the thermal simulation of a system with multiple chiplets,
bonding layers, TIM (Thermal Interface Material), and heatsink.

Function: simulator_simulate()
- Input: boxes (chiplets), bonding_box_list, TIM_boxes, heatsink_obj, layers
- Output: Dictionary mapping box names to (peak_temp, avg_temp, R_x, R_y, R_z)
"""

# ============================================================================
# THERMAL CONDUCTIVITY LOOKUP TABLE
# ============================================================================

conductivity_values = {
    "Air": 0.025,
    "FR-4": 0.1,
    "Cu-Foil": 400,
    "Si": 105,
    "Aluminium": 205,
    "TIM001": 100,
    "Glass": 1.36,
    "TIM": 100,
    "SnPb 67/37": 36,
    "Epoxy, Silver filled": 1.6,
    "SiO2": 1.1,
    "AlN": 237,
    "EpAg": 1.6,
    "Infill_material": 19,
    "Polymer1": 675,
    "TIM0p5": 1.0
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_layer_by_name(layers, layer_name):
    """
    Find a layer object by name from the layers list.
    
    Args:
        layers: List[Layer] - List of Layer objects from parse_Layer_netlist()
        layer_name: str - Name of the layer to find (e.g., "5nm_GPU_active")
    
    Returns:
        Layer object if found, None otherwise
    """
    if not layers:
        return None
    
    for layer in layers:
        if layer.get_name() == layer_name:
            return layer
    
    return None

def get_effective_conductivity(material_string, conductivity_values):
    """
    Calculate effective thermal conductivity for composite materials.
    
    Handles both simple and composite material definitions:
    - Simple: "Si" → single material
    - Composite: "Cu-Foil:0.5,Si:0.5" → 50% Cu-Foil, 50% Si
    
    For composite materials, uses the rule of mixtures (parallel resistors):
    1/k_eff = sum(fraction_i / k_i)
    
    This is the resistor model (worst case, conservative estimate).
    
    Args:
        material_string: str - Material definition (simple or composite)
        conductivity_values: Dict - Lookup table of k values [W/(m·K)]
    
    Returns:
        k_eff: float - Effective thermal conductivity [W/(m·K)]
    """
    
    if not material_string:
        print(f"Warning: Empty material string, using default Si conductivity")
        return conductivity_values.get("Si", 105)
    
    material_string = material_string.strip()
    
    # ---- Case 1: Simple material (no colon) ----
    if ":" not in material_string:
        material_name = material_string.strip()
        if material_name in conductivity_values:
            return conductivity_values[material_name]
        else:
            print(f"Warning: Material '{material_name}' not found, using default Si")
            return conductivity_values.get("Si", 105)
    
    # ---- Case 2: Composite material (contains colons) ----
    # Example: "Cu-Foil:0.5,Si:0.5"
    
    # Parse the composite definition
    components = material_string.split(",")
    
    total_fraction = 0.0
    inverse_k_sum = 0.0  # Sum for rule of mixtures: sum(f_i / k_i)
    
    for component in components:
        component = component.strip()
        
        # Split by colon
        parts = component.split(":")
        if len(parts) != 2:
            print(f"Warning: Invalid component format: '{component}'")
            continue
        
        material_name = parts[0].strip()
        fraction_str = parts[1].strip()
        
        # Parse fraction
        try:
            fraction = float(fraction_str)
        except ValueError:
            print(f"Warning: Could not parse fraction '{fraction_str}' for material '{material_name}'")
            continue
        
        # Look up conductivity
        if material_name not in conductivity_values:
            print(f"Warning: Material '{material_name}' not in conductivity lookup table")
            continue
        
        k = conductivity_values[material_name]
        
        if k <= 0:
            print(f"Warning: Material '{material_name}' has invalid conductivity: {k}")
            continue
        
        # Add to sum using rule of mixtures (resistor model)
        # For parallel resistors: 1/R_total = sum(f_i / R_i)
        # Which translates to: 1/k_eff = sum(f_i / k_i)
        inverse_k_sum += fraction / k
        total_fraction += fraction
    
    # Handle edge cases
    if total_fraction <= 0 or inverse_k_sum <= 0:
        print(f"Warning: Invalid composite material definition: '{material_string}'")
        return conductivity_values.get("Si", 105)
    
    # Normalize fractions and calculate effective conductivity
    # k_eff = total_fraction / inverse_k_sum
    k_eff = total_fraction / inverse_k_sum
    
    return k_eff


def calculate_stackup_resistance(box, layers, conductivity_values):
    """
    Calculate total thermal resistance of a box's stackup (vertical Z-direction).
    
    NOW SUPPORTS COMPOSITE MATERIALS!
    
    The stackup string defines layers and their properties. For example:
        "1:5nm_GPU_active,2:5nm_GPU_metal" means:
        - 1 repetition of layer "5nm_GPU_active"
        - 2 repetitions of layer "5nm_GPU_metal"
    
    Each layer can have:
    - Simple material: material="Si"
    - Composite material: material="Cu-Foil:0.5,Si:0.5"
    
    Args:
        box: Box object with stackup property
        layers: List[Layer] objects from parse_Layer_netlist()
        conductivity_values: Dict mapping material names to k values [W/(m·K)]
    
    Returns:
        R_z: float - Total thermal resistance [K/W]
    """
    if not box.stackup or box.stackup == "":
        return 0.0
    
    # Get box area for heat flow (X-Y plane)
    width_mm = box.width
    length_mm = box.length
    area_m2 = (width_mm / 1000) * (length_mm / 1000)  # Convert mm² to m²
    
    if area_m2 <= 0:
        return 0.0
    
    total_resistance = 0.0
    
    # Parse stackup string: "1:layer1,2:layer2,..." → [("1", "layer1"), ("2", "layer2"), ...]
    layer_specs = box.stackup.split(",")
    
    for layer_spec in layer_specs:
        layer_spec = layer_spec.strip()
        
        # Split by colon to get count and layer name
        parts = layer_spec.split(":")
        if len(parts) != 2:
            print(f"Warning: Invalid layer spec format: '{layer_spec}' in box {box.name}")
            continue
        
        count_str, layer_name = parts
        count_str = count_str.strip()
        layer_name = layer_name.strip()
        
        # Parse count
        try:
            count = int(count_str)
        except ValueError:
            print(f"Warning: Could not parse count '{count_str}' in box {box.name}")
            continue
        
        # Find layer object
        layer_obj = find_layer_by_name(layers, layer_name)
        if layer_obj is None:
            print(f"Warning: Layer '{layer_name}' not found in layer database for box {box.name}")
            continue
        
        # Get material and thickness
        material_string = layer_obj.get_material()
        thickness_um = layer_obj.get_thickness()
        
        if thickness_um is None or thickness_um <= 0:
            print(f"Warning: Layer '{layer_name}' has invalid thickness: {thickness_um}")
            continue
        
        # Convert thickness from micrometers to meters
        thickness_m = thickness_um * 1e-6
        
        # ---- NEW: Get effective conductivity for composite materials ----
        k_eff = get_effective_conductivity(material_string, conductivity_values)
        
        if k_eff <= 0:
            print(f"Warning: Layer '{layer_name}' has invalid effective conductivity: {k_eff}")
            continue
        
        # Calculate resistance for this layer: R = t / (k * A)
        R_layer = thickness_m / (k_eff * area_m2)
        
        # Account for multiple repetitions
        total_resistance += count * R_layer
    
    return total_resistance

def calculate_stackup_resistance(box, layers, conductivity_values):
    """
    Calculate total thermal resistance of a box's stackup (vertical Z-direction).
    
    The stackup string defines layers and their properties. For example:
        "1:5nm_GPU_active,2:5nm_GPU_metal" means:
        - 1 repetition of layer "5nm_GPU_active"
        - 2 repetitions of layer "5nm_GPU_metal"
    
    Args:
        box: Box object with stackup property
        layers: List[Layer] objects from parse_Layer_netlist()
        conductivity_values: Dict mapping material names to k values [W/(m·K)]
    
    Returns:
        R_z: float - Total thermal resistance [K/W]
    """
    if not box.stackup or box.stackup == "":
        return 0.0
    
    # Get box area for heat flow (X-Y plane)
    width_mm = box.width
    length_mm = box.length
    area_m2 = (width_mm / 1000) * (length_mm / 1000)  # Convert mm² to m²
    
    if area_m2 <= 0:
        return 0.0
    
    total_resistance = 0.0
    
    # Parse stackup string: "1:layer1,2:layer2,..." → [("1", "layer1"), ("2", "layer2"), ...]
    layer_specs = box.stackup.split(",")
    
    for layer_spec in layer_specs:
        layer_spec = layer_spec.strip()
        
        # Split by colon to get count and layer name
        parts = layer_spec.split(":")
        if len(parts) != 2:
            print(f"Warning: Invalid layer spec format: '{layer_spec}' in box {box.name}")
            continue
        
        count_str, layer_name = parts
        count_str = count_str.strip()
        layer_name = layer_name.strip()
        
        # Parse count
        try:
            count = int(count_str)
        except ValueError:
            print(f"Warning: Could not parse count '{count_str}' in box {box.name}")
            continue
        
        # Find layer object
        layer_obj = find_layer_by_name(layers, layer_name)
        if layer_obj is None:
            print(f"Warning: Layer '{layer_name}' not found in layer database for box {box.name}")
            continue
        
        # Get material and thickness
        material = layer_obj.get_material()
        thickness_um = layer_obj.get_thickness()
        
        if thickness_um is None or thickness_um <= 0:
            print(f"Warning: Layer '{layer_name}' has invalid thickness: {thickness_um}")
            continue
        
        # Convert thickness from micrometers to meters
        thickness_m = thickness_um * 1e-6
        
        # Get thermal conductivity
        if material not in conductivity_values:
            print(f"Warning: Material '{material}' not in conductivity lookup table")
            continue
        
        k = conductivity_values[material]
        
        if k <= 0:
            print(f"Warning: Material '{material}' has invalid conductivity: {k}")
            continue
        
        # Calculate resistance for this layer: R = t / (k * A)
        R_layer = thickness_m / (k * area_m2)
        
        # Account for multiple repetitions
        total_resistance += count * R_layer
    
    return total_resistance


def calculate_box_resistances(box, layers, conductivity_values):
    """
    Calculate thermal resistances in X, Y, and Z directions for a box.
    
    The box is modeled as a rectangular conductor with:
    - R_x: resistance for heat flowing in X direction (through width)
    - R_y: resistance for heat flowing in Y direction (through length)
    - R_z: resistance for heat flowing in Z direction (through stackup)
    
    Args:
        box: Box object
        layers: List[Layer] objects
        conductivity_values: Dict of material conductivities
    
    Returns:
        (R_x, R_y, R_z): tuple of float - Resistances [K/W]
    """
    
    # Get box dimensions in meters
    width_m = box.width / 1000
    length_m = box.length / 1000
    height_m = box.height / 1000
    
    # For lateral heat flow, use silicon conductivity (main material)
    k_lateral = conductivity_values.get("Si", 105)  # Silicon: 105 W/(m·K)
    
    # --- R_X: Heat flowing in X direction ---
    # Heat travels through WIDTH dimension
    # Perpendicular area = length × height
    area_yz = length_m * height_m
    if area_yz > 0:
        R_x = width_m / (k_lateral * area_yz)
    else:
        R_x = 0.0
    
    # ---- R_Y: Heat flowing in Y direction ---
    # Heat travels through LENGTH dimension
    # Perpendicular area = width × height
    area_xz = width_m * height_m
    if area_xz > 0:
        R_y = length_m / (k_lateral * area_xz)
    else:
        R_y = 0.0
    
    # ---- R_Z: Heat flowing in Z direction (vertical through stackup) ---
    R_z = calculate_stackup_resistance(box, layers, conductivity_values)
    
    return R_x, R_y, R_z


def calculate_total_path_resistance(box, resistances_dict, bonding_box_list, TIM_boxes, heatsink_obj):
    """
    Calculate total thermal resistance path from a box to ambient.
    
    Path: Box stackup → Bonding layer → TIM → Heatsink → Ambient
    
    Args:
        box: Box object (chiplet)
        resistances_dict: Dict mapping box names to (R_x, R_y, R_z)
        bonding_box_list: List of bonding Box objects
        TIM_boxes: List of TIM Box objects
        heatsink_obj: Dict with heatsink properties
    
    Returns:
        R_total: float - Total resistance [K/W]
    """
    
    # Start with the box's own Z-direction resistance (stackup)
    R_total = resistances_dict[box.name][2]  # R_z of the box itself
    
    # ---- Add bonding layer resistance ----
    for bonding_box in bonding_box_list:
        # Check if this bonding box is associated with current box
        if bonding_box.name.startswith(box.name) and "bonding" in bonding_box.name:
            R_bonding = resistances_dict.get(bonding_box.name, (0, 0, 0))[2]  # R_z
            R_total += R_bonding
    
    # ---- Add TIM resistance ----
    for tim_box in TIM_boxes:
        # Check if this TIM box is associated with current box
        if tim_box.name.startswith(box.name) and "TIM" in tim_box.name:
            R_tim = resistances_dict.get(tim_box.name, (0, 0, 0))[2]  # R_z
            R_total += R_tim
    
    # ---- Add heatsink resistance ----
    if heatsink_obj:
        try:
            hc = float(heatsink_obj.get("hc", 10000))  # Heat transfer coefficient [W/(m²·K)]
            dx_mm = float(heatsink_obj.get("base_dx", 30))  # Width [mm]
            dy_mm = float(heatsink_obj.get("base_dy", 30))  # Length [mm]
            
            # Convert to meters
            dx_m = dx_mm / 1000
            dy_m = dy_mm / 1000
            
            area_m2 = dx_m * dy_m
            
            if area_m2 > 0 and hc > 0:
                R_heatsink = 1.0 / (hc * area_m2)
                R_total += R_heatsink
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse heatsink parameters: {e}")
    
    return R_total


def calculate_box_temperatures(box, all_boxes, bonding_box_list, TIM_boxes, 
                               heatsink_obj, resistances_dict):
    """
    Calculate peak and average temperatures for a box.
    
    Temperature calculation approach:
    1. For boxes with power (chiplets): T = T_ambient + power * R_total
    2. For boxes without power: Use a simplified approach
    
    Note: This is a simplified model. A more sophisticated approach would solve
    the full thermal network using Kirchhoff's laws.
    
    Args:
        box: Box object
        all_boxes: List of all boxes (for reference)
        bonding_box_list: List of bonding boxes
        TIM_boxes: List of TIM boxes
        heatsink_obj: Dict with heatsink properties
        resistances_dict: Dict mapping box names to (R_x, R_y, R_z)
    
    Returns:
        (T_peak, T_avg): tuple of float - Peak and average temperatures [°C]
    """
    
    T_ambient = 25.0  # Ambient temperature [°C]
    
    # ---- Case 1: Box has power (chiplet) ----
    if box.power and box.power > 0:
        # Calculate total resistance path to ambient
        R_total = calculate_total_path_resistance(box, resistances_dict, bonding_box_list, 
                                                   TIM_boxes, heatsink_obj)
        
        # Temperature rise above ambient: ΔT = P * R
        delta_T = box.power * R_total
        
        T_peak = T_ambient + delta_T
        T_avg = T_ambient + delta_T / 2
    
    # ---- Case 2: Box has no power (bonding, TIM, etc.) ----
    else:
        # For non-power-dissipating layers, set to ambient
        # (More sophisticated models would interpolate between adjacent layers)
        T_peak = T_ambient
        T_avg = T_ambient
    
    return T_peak, T_avg


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def simulator_simulate(boxes, bonding_box_list, TIM_boxes, heatsink_obj=None, 
                       heatsink_list=None, heatsink_name=None, bonding_list=None,
                       bonding_name_type_dict=None, is_repeat=False, 
                       min_TIM_height=0.1, power_dict=None, layers=None):
    """
    Simulate thermal behavior of a 3D/2.5D system.
    
    This function computes thermal resistances and temperatures for all boxes
    (chiplets, bonding layers, TIM layers) in the system.
    
    INPUT PARAMETERS:
    -----------------
    boxes: List[Box]
        Main chiplet boxes (GPU, HBM, interposer, etc.)
        Each has: name, width, length, height, power, stackup, chiplet_parent
    
    bonding_box_list: List[Box]
        Thin bonding layers between chiplets
        Created by create_all_bonding()
    
    TIM_boxes: List[Box]
        Thermal Interface Material layers
        Created by create_TIM_to_heatsink()
    
    heatsink_obj: Dict
        Heatsink properties (hc, base_dx, base_dy, etc.)
        Created by create_heat_sink()
    
    layers: List[Layer]
        Layer definitions (thickness, material, conductivity)
        Parsed from layer_definitions.xml
    
    [Other parameters currently unused in basic implementation]
    
    RETURN VALUE:
    -------------
    results: Dict
        Dictionary mapping box names to thermal data:
        {
            "BoxName": (peak_temp, avg_temp, R_x, R_y, R_z),
            ...
        }
        
        Where:
        - peak_temp: float [°C] - Highest temperature in box
        - avg_temp: float [°C] - Average temperature in box
        - R_x: float [K/W] - Thermal resistance in X direction
        - R_y: float [K/W] - Thermal resistance in Y direction
        - R_z: float [K/W] - Thermal resistance in Z direction
    
    EXAMPLE OUTPUT:
    ---------------
    {
        "GPU#0": (35.2, 30.1, 1.9, 1.9, 0.0000005),
        "HBM_l1#0": (33.5, 29.2, 1.8, 1.8, 0.0001),
        "GPU#0_bonding": (34.9, 32.5, 0.001, 0.001, 0.0005),
        "GPU#0_TIM": (28.5, 27.0, 0.1, 0.1, 3.0),
        "interposer": (25.5, 25.1, 100, 100, 0.05),
        ...
    }
    """
    
    results = {}
    
    # ========================================================================
    # STEP 1: Combine all boxes
    # ========================================================================
    all_boxes = boxes + bonding_box_list + TIM_boxes
    
    print(f"[Simulator] Starting thermal simulation with {len(all_boxes)} total boxes")
    print(f"  - Main chiplets: {len(boxes)}")
    print(f"  - Bonding layers: {len(bonding_box_list)}")
    print(f"  - TIM layers: {len(TIM_boxes)}")
    
    # ========================================================================
    # STEP 2: Validate inputs
    # ========================================================================
    if not all_boxes:
        print("[ERROR] No boxes found in input!")
        return results
    
    if not layers:
        print("[ERROR] No layer definitions provided!")
        return results
    
    # ========================================================================
    # STEP 3: Calculate thermal resistances for all boxes
    # ========================================================================
    print("\n[Step 1] Calculating thermal resistances...")
    
    resistances_dict = {}
    
    for box in all_boxes:
        try:
            R_x, R_y, R_z = calculate_box_resistances(box, layers, conductivity_values)
            resistances_dict[box.name] = (R_x, R_y, R_z)
            
            if box.power > 0:
                print(f"  {box.name:25} - R: ({R_x:.6e}, {R_y:.6e}, {R_z:.6e}) K/W - Power: {box.power:.1f}W")
            else:
                print(f"  {box.name:25} - R: ({R_x:.6e}, {R_y:.6e}, {R_z:.6e}) K/W")
        
        except Exception as e:
            print(f"[ERROR] Failed to calculate resistance for {box.name}: {e}")
            # Use zero resistance as fallback
            resistances_dict[box.name] = (0.0, 0.0, 0.0)
    
    # ========================================================================
    # STEP 4: Calculate temperatures for all boxes
    # ========================================================================
    print("\n[Step 2] Calculating temperatures...")
    
    for box in all_boxes:
        try:
            T_peak, T_avg = calculate_box_temperatures(box, all_boxes, bonding_box_list, 
                                                       TIM_boxes, heatsink_obj, 
                                                       resistances_dict)
            
            R_x, R_y, R_z = resistances_dict[box.name]
            
            # Store result
            results[box.name] = (T_peak, T_avg, R_x, R_y, R_z)
            
            if box.power > 0:
                print(f"  {box.name:25} - T: {T_peak:.2f}°C (peak), {T_avg:.2f}°C (avg) - Power: {box.power:.1f}W")
            else:
                print(f"  {box.name:25} - T: {T_peak:.2f}°C (peak), {T_avg:.2f}°C (avg)")
        
        except Exception as e:
            print(f"[ERROR] Failed to calculate temperature for {box.name}: {e}")
            # Use ambient as fallback
            R_x, R_y, R_z = resistances_dict.get(box.name, (0, 0, 0))
            results[box.name] = (25.0, 25.0, R_x, R_y, R_z)
    
    # ========================================================================
    # STEP 5: Summary
    # ========================================================================
    print("\n[Summary]")
    print(f"  Total boxes processed: {len(results)}")
    
    # Find peak temperature
    peak_temps = [result[0] for result in results.values()]
    if peak_temps:
        max_temp = max(peak_temps)
        max_box = [name for name, (t, _, _, _, _) in results.items() if t == max_temp][0]
        print(f"  Maximum temperature: {max_temp:.2f}°C in box '{max_box}'")
    
    print("\n[Simulation Complete]")
    
    return results


# ============================================================================
# STANDALONE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of simulator_simulate()
    
    To use this:
    1. Import layer definitions: layers = parse_Layer_netlist(...)
    2. Create boxes and layers: boxes, bonding_box_list, TIM_boxes = ...
    3. Create heatsink: heatsink_obj = create_heat_sink(...)
    4. Run simulation: results = simulator_simulate(boxes, bonding_box_list, TIM_boxes, ...)
    """
    
    print("This module implements simulator_simulate() for thermal simulation.")
    print("See therm.py for usage within the main thermal analysis pipeline.")

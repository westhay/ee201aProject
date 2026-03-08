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
    Parallel rule of mixtures: k_eff = Σ(f_i * k_i)
    Used for: Composite materials where components are mixed side-by-side.
    """
    if not material_string:
        return conductivity_values.get("Si", 105)
    
    material_string = material_string.strip()
    
    if ":" not in material_string:
        return conductivity_values.get(material_string, 105)
    
    components = material_string.split(",")
    k_eff = 0.0
    total_fraction = 0.0
    
    for component in components:
        component = component.strip()
        parts = component.split(":")
        if len(parts) != 2:
            continue
        
        material_name = parts[0].strip()
        try:
            fraction = float(parts[1].strip())
        except ValueError:
            continue
        
        if material_name not in conductivity_values:
            print(f"Warning: Material '{material_name}' not found")
            continue
        
        k = conductivity_values[material_name]
        if k <= 0:
            continue
        
        k_eff += fraction * k
        total_fraction += fraction
    
    if total_fraction <= 0:
        return conductivity_values.get("Si", 105)
    
    return k_eff

def parse_stackup(stackup_string):
    if not stackup_string or stackup_string == "":
        return []
    
    result = []
    current_pos = 0
    
    while current_pos < len(stackup_string):
        # Find the next colon (separates count from material)
        colon_pos = stackup_string.find(":", current_pos)
        
        if colon_pos == -1:
            # No more colons, malformed
            break
        
        # Extract count (everything from current_pos to colon)
        count_str = stackup_string[current_pos:colon_pos].strip()
        
        try:
            count = int(count_str)
        except ValueError:
            print(f"Warning: Could not parse count: '{count_str}'")
            break
        
        # Now find where this material definition ends
        # It ends at the next "count:" pattern (digit(s) followed by colon)
        # OR at the end of the string
        
        material_start = colon_pos + 1
        material_end = len(stackup_string)
        
        # Look for the next "count:" pattern
        for i in range(material_start, len(stackup_string)):
            if stackup_string[i] == ',':
                # Check if what follows is a count (digits followed by colon)
                remaining = stackup_string[i+1:].lstrip()
                
                # Try to parse as count:
                colon_in_remaining = remaining.find(":")
                if colon_in_remaining > 0:
                    potential_count = remaining[:colon_in_remaining].strip()
                    
                    # Check if it's all digits (a valid count)
                    if potential_count.isdigit():
                        # Found the next count, so material ends at this comma
                        material_end = i
                        break
        
        # Extract material definition
        material_def = stackup_string[material_start:material_end].strip()
        
        if material_def:
            result.append((count, material_def))
        
        # Move to next position
        current_pos = material_end
        if current_pos < len(stackup_string) and stackup_string[current_pos] == ',':
            current_pos += 1  # Skip the comma
    
    return result
    
def calculate_single_layer_resistances_special(box, conductivity_values):
    """
    Special handler for TIM and bonding boxes.
    
    NOW HANDLES MATERIALS WITH COMMAS IN THEIR NAMES!
    
    Args:
        box: Box object
        conductivity_values: Dict of k values
    
    Returns:
        (R_x, R_y, R_z): tuple [K/W]
    """
    
    if not box.stackup or box.stackup == "":
        return (0.0, 0.0, 0.0)
    
    # Parse stackup carefully
    layer_specs = parse_stackup(box.stackup)
    
    if not layer_specs:
        return (0.0, 0.0, 0.0)
    
    R_x_list = []
    R_y_list = []
    R_z_list = []
    
    for count, material_def in layer_specs:
        # material_def might be:
        # - "TIM0p5" (simple)
        # - "EpAg:75,Epoxy, Silver filled:25" (composite with comma in name)
        # - "Cu-Foil:0.5,Si:0.5" (composite without comma in names)
        
        # Get effective conductivity (handles composites)
        k_eff = get_effective_conductivity(material_def, conductivity_values)
        
        if k_eff <= 0:
            continue
        
        # Calculate resistances
        height_m = box.height / 1000
        width_m = box.width / 1000
        length_m = box.length / 1000
        
        area_xy = width_m * length_m
        area_yz = length_m * height_m
        area_xz = width_m * height_m
        
        if area_yz > 0:
            R_x = width_m / (k_eff * area_yz)
        else:
            R_x = 0.0
        
        if area_xz > 0:
            R_y = length_m / (k_eff * area_xz)
        else:
            R_y = 0.0
        
        if area_xy > 0:
            R_z = height_m / (k_eff * area_xy)
        else:
            R_z = 0.0
        
        for _ in range(count):
            R_x_list.append(R_x)
            R_y_list.append(R_y)
            R_z_list.append(R_z)
    
    R_x_total = combine_parallel_resistances(R_x_list)
    R_y_total = combine_parallel_resistances(R_y_list)
    R_z_total = combine_series_resistances(R_z_list)
    
    return (R_x_total, R_y_total, R_z_total)

def calculate_single_layer_resistances(layer_obj, box, conductivity_values):
    """
    Calculate R_x, R_y, R_z for a SINGLE layer.
    
    Uses the parallel rule of mixtures for composite materials.
    
    Args:
        layer_obj: Layer object with get_material() and get_thickness()
        box: Box object with width, length, height
        conductivity_values: Dict of k values
    
    Returns:
        (R_x, R_y, R_z): tuple of float [K/W]
    """
    
    # Get layer properties
    material_string = layer_obj.get_material()
    thickness_um = layer_obj.get_thickness()
    
    if thickness_um is None or thickness_um <= 0:
        return (0.0, 0.0, 0.0)
    
    # Convert to meters
    thickness_m = thickness_um * 1e-6
    box_width_m = box.width / 1000
    box_length_m = box.length / 1000
    box_height_m = box.height / 1000
    
    # Calculate effective conductivity (parallel for composite)
    k_eff = get_effective_conductivity(material_string, conductivity_values)
    
    if k_eff <= 0:
        return (0.0, 0.0, 0.0)
    
    # Box area
    area_xy = box_width_m * box_length_m  # For R_z
    area_yz = box_length_m * thickness_m  # For R_x
    area_xz = box_width_m * thickness_m   # For R_y
    
    # Calculate resistances for this layer
    # R_x: heat flowing through width
    if area_yz > 0:
        R_x = box_width_m / (k_eff * area_yz)
    else:
        R_x = 0.0
    
    # R_y: heat flowing through length
    if area_xz > 0:
        R_y = box_length_m / (k_eff * area_xz)
    else:
        R_y = 0.0
    
    # R_z: heat flowing through thickness
    if area_xy > 0:
        R_z = thickness_m / (k_eff * area_xy)
    else:
        R_z = 0.0
    
    return (R_x, R_y, R_z)


def combine_parallel_resistances(resistances_list):
    """
    Combine resistances in PARALLEL: 1/R_total = Σ(1/R_i)
    
    Args:
        resistances_list: List of resistance values [K/W]
    
    Returns:
        R_total: float [K/W]
    """
    if not resistances_list or len(resistances_list) == 0:
        return 0.0
    
    inverse_sum = sum(1/r for r in resistances_list if r > 0)
    
    if inverse_sum <= 0:
        return 0.0
    
    return 1 / inverse_sum


def combine_series_resistances(resistances_list):
    """
    Combine resistances in SERIES: R_total = Σ(R_i)
    
    Args:
        resistances_list: List of resistance values [K/W]
    
    Returns:
        R_total: float [K/W]
    """
    return sum(r for r in resistances_list if r >= 0)


def calculate_box_resistances(box, layers, conductivity_values):
    """
    Calculate total R_x, R_y, R_z for a box's stackup.
    
    APPROACH:
    1. For each layer in stackup: calculate individual R_x, R_y, R_z
    2. Combine horizontally (R_x, R_y) in PARALLEL
    3. Combine vertically (R_z) in SERIES
    
    Args:
        box: Box object with stackup property
        layers: List[Layer] objects
        conductivity_values: Dict of k values
    
    Returns:
        (R_x_total, R_y_total, R_z_total): tuple [K/W]
    """
    is_tim_box = "_TIM" in box.name
    is_bonding_box = "_bonding" in box.name
    
    if is_tim_box or is_bonding_box:
        # ---- Case 1: TIM or Bonding Box ----
        # Use special handler that works with direct material names
        return calculate_single_layer_resistances_special(box, conductivity_values)    
    else:
        if not box.stackup or box.stackup == "":
            return (0.0, 0.0, 0.0)
        
        # Parse stackup: "1:layer1,2:layer2"
        layer_specs = box.stackup.split(",")
        
        R_x_list = []  # For parallel combination
        R_y_list = []  # For parallel combination
        R_z_list = []  # For series combination
        
        for layer_spec in layer_specs:
            layer_spec = layer_spec.strip()
            parts = layer_spec.split(":")
            
            if len(parts) != 2:
                print(f"Warning: Invalid layer spec: {layer_spec}")
                continue
            
            count_str, layer_name = parts
            
            try:
                count = int(count_str.strip())
            except ValueError:
                print(f"Warning: Could not parse count: {count_str}")
                continue
            
            # Find layer object
            layer_obj = find_layer_by_name(layers, layer_name.strip())
            if layer_obj is None:
                print(f"Warning: Layer '{layer_name}' not found")
                continue
            
            # Calculate R_x, R_y, R_z for this layer
            R_x_layer, R_y_layer, R_z_layer = calculate_single_layer_resistances(
                layer_obj, box, conductivity_values
            )
            
            # Account for repetitions
            for _ in range(count):
                R_x_list.append(R_x_layer)
                R_y_list.append(R_y_layer)
                R_z_list.append(R_z_layer)
    
    # Combine resistances
    # Lateral (X, Y): Layers in PARALLEL
    R_x_total = combine_parallel_resistances(R_x_list)
    R_y_total = combine_parallel_resistances(R_y_list)
    
    # Vertical (Z): Layers in SERIES
    R_z_total = combine_series_resistances(R_z_list)
    
    return (R_x_total, R_y_total, R_z_total)

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

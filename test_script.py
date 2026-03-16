"""Quick test for thermal_grid.py - Save as test_thermal_grid.py"""

import numpy as np
import pandas as pd
from thermal_grid import create_voxel_grid, calculate_voxel_resistances, solve_temperature_grid, summarize_temperature_grid, write_temperature_report, build_results_dict_per_project_requirements, write_box_results_report
from rearrange import Box

# Conductivity values
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
# Create simple test boxes

def simulator_simulate(boxes, bonding_box_list, TIM_boxes, heatsink_obj, 
                       heatsink_list, heatsink_name, bonding_list,
                       bonding_name_type_dict, is_repeat, 
                       min_TIM_height, power_dict, anemoi_parameter_ID,
                       layers):
    all_boxes = boxes + bonding_box_list+ TIM_boxes
                           
    print("Creating voxel grid...")
    grid_info = create_voxel_grid(all_boxes, voxel_size=0.5, layers = layers,
                                  conductivity_values=conductivity_values)
    
    print("Calculating resistances...")
    resistance_grid = calculate_voxel_resistances(grid_info)
    
    print("\nExporting to CSV...")
    
    # 1. Grid summary
    nx, ny, nz = grid_info['grid_shape']
    pd.DataFrame([{
        'nx': nx, 'ny': ny, 'nz': nz,
        'total_voxels': nx*ny*nz,
        'voxel_size': grid_info["voxel_size"],
        'bounds': str(grid_info['bounds'])
    }]).to_csv('grid_summary.csv', index=False)
    
    # 2. Material distribution
    unique, counts = np.unique(grid_info['material_grid'], return_counts=True)
    pd.DataFrame({
        'material': unique,
        'count': counts,
        'percentage': 100*counts/(nx*ny*nz)
    }).to_csv('material_distribution.csv', index=False)
    
    # 3. Voxel sample (every 10th)
    samples = []
    for i in range(0, nx, 10):
        for j in range(0, ny, 10):
            for k in range(0, nz, 10):
                samples.append({
                    'i': i, 'j': j, 'k': k,
                    'material': grid_info['material_grid'][i,j,k],
                    'k_W_per_mK': grid_info['conductivity_grid'][i,j,k],
                    'power_W_per_m3': grid_info['power_grid'][i,j,k],
                    'box': grid_info['box_grid'][i,j,k],
                    'R_x': resistance_grid[i,j,k,0],
                    'R_y': resistance_grid[i,j,k,1],
                    'R_z': resistance_grid[i,j,k,2],
                })
    pd.DataFrame(samples).to_csv('voxel_sample.csv', index=False)

    print("✓ Created grid_summary.csv")
    print("✓ Created material_distribution.csv")
    print(f"✓ Created voxel_sample.csv ({len(samples)} samples)")
    print("\nDone! Open CSV files to inspect results.")

    temperature_grid, circuit, analysis = solve_temperature_grid(
        conductivity_grid=grid_info["conductivity_grid"],
        power_grid=grid_info["power_grid"],
        voxel_size_mm=grid_info["voxel_size"],
        T_ambient=25.0,
        h_top=1000.0,
        h_side=10.0,
        h_bottom=100.0,
        active_mask=grid_info.get("active_mask", None)
    )

    summary = summarize_temperature_grid(
        temperature_grid,
        active_mask=grid_info.get("active_mask", None),
        voxel_size_mm=grid_info["voxel_size"],
        bounds=grid_info["bounds"],
        T_ambient=25.0
    )

    print("Temperature summary:")
    print(summary)
    
    write_temperature_report(summary, "temperature_summary.txt")

    print("Temperature summary:")
    print(summary)

    results = build_results_dict_per_project_requirements(
    temperature_grid=temperature_grid,
    resistance_grid=resistance_grid,
    box_grid=grid_info["box_grid"],
    active_mask=grid_info.get("active_mask", None),
    )
    write_box_results_report(results, "box_results.txt")
    return results
    

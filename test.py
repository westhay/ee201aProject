"""Quick test for thermal_grid.py - Save as test_thermal_grid.py"""

import numpy as np
import pandas as pd
from thermal_grid import create_voxel_grid, calculate_voxel_resistances
from rearrange import Box

# Conductivity values
conductivity_values = {
    "Air": 0.025, "Si": 105, "Cu-Foil": 400, "EpAg": 1.6, "TIM": 100
}

# Create simple test boxes
boxes = [
    Box(0, 0, 0, 30, 30, 0.5, 0, 'Si', 0, 'interposer'),
    Box(2, 0, 0.5, 26, 32, 0.7, 270, 'Si', 0, 'GPU#0'),
    Box(0, 0, 0.5, 7, 11, 0.7, 5, 'Si', 0, 'HBM#0'),
    Box(2, 0, 0.48, 26, 32, 0.02, 0, '1:Cu-Foil:70.0,EpAg:30.0', 0, 'bonding'),
]

print("Creating voxel grid...")
grid_info = create_voxel_grid(boxes, voxel_size=0.1, 
                              conductivity_values=conductivity_values)

print("Calculating resistances...")
resistance_grid = calculate_voxel_resistances(grid_info)

print("\nExporting to CSV...")

# 1. Grid summary
nx, ny, nz = grid_info['grid_shape']
pd.DataFrame([{
    'nx': nx, 'ny': ny, 'nz': nz,
    'total_voxels': nx*ny*nz,
    'voxel_size': 0.1,
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

"""
Multi-objective optimization: topographic modification optimization
Author: Hanwen Xu
Version: 1
Date: Dec 10, 2023
demo version for AR-Sandbox combined MOO assistance, you can install Sandworm plug-in in
Rhino+Grasshopper platform to run it.

Pre-evaluation of multi-index assessment
"""

import whitebox_workflows as wbw
import numpy as np
import pandas as pd
import rasterio as rs
from rasterio.plot import show
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_file_path = r'D:\PhD career\08 Conference and activity\07 DLA Conference\AR_Sandbox_MOO\00_data_source\50x50.csv'
df = pd.read_csv(csv_file_path, header=None)

# Convert the DataFrame to a list of lists (each row becomes a list)
data_array = df.values
data_list = data_array.flatten().tolist()
print(data_list)

# create DEM with elevation value of the list
wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\08 Conference and activity\07 DLA Conference\AR_Sandbox_MOO\00_data_source'
dem = wbe.read_raster('Blank_DEM_clip50.tif')

i = 0
for row in range(dem.configs.rows):
    for col in range(dem.configs.columns):
        if i < len(data_list):
            dem[row, col] = data_list[i]
            i += 1

wbe.write_raster(dem, file_name='Initialization.tif', compress=True)
path_00 = '../00_data_source/Initialization.tif'
image_00 = rs.open(path_00)
fig, ax = plt.subplots(figsize=(16, 16))
ax.tick_params(axis='both', which='major', labelsize=20)
show(image_00, title='DEM', ax=ax)

plt.ticklabel_format(style='plain')
plt.show()



### --------------------------------------------------- ###
# Pre_analysis
# Path_length
flow_accum = wbe.d8_flow_accum(dem, out_type='cells')
Flow_accum_value = []
for row in range(flow_accum.configs.rows):
    for col in range(flow_accum.configs.columns):
        elev = flow_accum[row, col]
        if elev != flow_accum.configs.nodata:
            Flow_accum_value.append(elev)

# print(Flow_accum_value)
print(max(Flow_accum_value))
print(min(Flow_accum_value))

V_threshold = max(Flow_accum_value) * 0.05

path_length = wbe.new_raster(flow_accum.configs)

for row in range(flow_accum.configs.rows):
    for col in range(flow_accum.configs.columns):
        elev = flow_accum[row, col]   # Read a cell value from a Raster
        if elev >= V_threshold and elev != flow_accum.configs.nodata:
            path_length[row, col] = 1.0   # Write the cell value of a Raster

        elif elev < V_threshold and elev != flow_accum.configs.nodata:
            path_length[row, col] = 0.0

        elif elev == flow_accum.configs.nodata:
            path_length[row, col] = flow_accum.configs.nodata

wbe.write_raster(path_length, 'DEM_path.tif', compress=True)


# visualization
path_01 = '../00_data_source/DEM_path.tif'
image_01 = rs.open(path_01)

fig, ax = plt.subplots(figsize=(16, 16))
ax.tick_params(axis='both', which='major', labelsize=20)
show(image_01, title='DEM_path', ax=ax)

plt.ticklabel_format(style='plain')
plt.show()


# value
Path_value = []
for row in range(path_length.configs.rows):
    for col in range(path_length.configs.columns):
        elev = path_length[row, col]
        if elev != path_length.configs.nodata:
            Path_value.append(elev)
print(sum(Path_value))



# Pre_analysis
# Velocity
slope = wbe.slope(dem, units="percent")
velocity = wbe.new_raster(slope.configs)

for row in range(slope.configs.rows):
    for col in range(slope.configs.columns):
        elev = slope[row, col]
        if elev == slope.configs.nodata:
            velocity[row, col] = slope.configs.nodata

        elif elev != slope.configs.nodata:
            slope_factor = (slope[row, col] / 100) ** 0.5
            flow_factor = (flow_accum[row, col] * 100 * 0.000004215717) ** (2 / 3)   # rainfall tensity: 0.000004215717
            velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

wbe.write_raster(velocity, 'DEM_velocity.tif', compress=True)

# visualization
path_02 = '../00_data_source/DEM_velocity.tif'
image_02 = rs.open(path_02)

fig, ax = plt.subplots(figsize=(16, 16))
ax.tick_params(axis='both', which='major', labelsize=20)
show(image_02, cmap='Blues', title='DEM_velocity', ax=ax)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.19, 0.03, 0.3])  # 调整颜色条的位置和大小
cbar = plt.colorbar(ax.images[0], cax=cbar_ax)
cbar.ax.tick_params(labelsize=20)

plt.ticklabel_format(style='plain')
ax.get_yaxis().get_major_formatter().set_scientific(False)  # 关闭科学计数法
plt.show()


# value
Velocity_value = []
for row in range(velocity.configs.rows):
    for col in range(velocity.configs.columns):
        elev = velocity[row, col]
        if elev != flow_accum.configs.nodata:
            Velocity_value.append(elev)

# print(Velocity_value)
print(max(Velocity_value))
print(min(Velocity_value))
print(np.median(Velocity_value))
"""
Multi-objective optimization: topographic modification optimization
Author: Hanwen Xu
Version: 1
Date: Dec 10, 2023
demo version for AR-Sandbox combined MOO assistance, you can install Sandworm plug-in in
Rhino+Grasshopper platform to run it.

AR-Sandbox freehand terrain -- Rhino mesh surface -- DEM --- elevation value list
Use the list as the beginning also as variables of this optimization process
"""

import whitebox_workflows as wbw
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import pandas as pd
import rasterio as rs
from rasterio.plot import show

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
print(show(image_00))

# setting the number of variables in opitimization
n_grid = len(data_list)
cut_and_fill = wbe.new_raster(dem.configs)


### -------------------------------------------------------- ###
# define MOO problem
class MyProblem(ElementwiseProblem):

    def __init__(self, n_grid, **kwargs):
        super().__init__(n_var=int(n_grid),
                         n_obj=3,
                         n_ieq_constr=1,
                         xl=np.array([-1] * n_grid),
                         xu=np.array([1] * n_grid),
                         **kwargs)
        self.n_grid = n_grid

    def _evaluate(self, x, out, *args, **kwargs):

        var_list = []
        for i in range(n_grid):
            var_list.append(x[i])

        # notice your function should be Min function
        earth_volume_function = sum(abs(i) for i in var_list)  # unit area of grid: input yours (m2)
        flow_length_function = - (path_sum_calculation(var_list))
        velocity_function = velocity_calculation(var_list)

        # notice your function should be <= 0
        g1 = sum(abs(i) for i in var_list) - 1250   # the limitation of total earthwork volume
        # g2 = - (path_sum_calculation(var_list)) + 193    # the original flow path lenth

        out["F"] = [earth_volume_function, flow_length_function, velocity_function]
        out["G"] = [g1]

def path_sum_calculation(var_list):
    i = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                cut_and_fill[row, col] = var_list[i]
                i = i + 1

    # creat dem_pop
    dem_pop = wbe.raster_calculator(expression="'dem' - 'cut_and _fill'", input_rasters=[dem, cut_and_fill])

    # path length calculation
    flow_accum = wbe.d8_flow_accum(dem_pop, out_type='cells')
    path_length = wbe.new_raster(flow_accum.configs)

    for row in range(flow_accum.configs.rows):
        for col in range(flow_accum.configs.columns):
            elev = flow_accum[row, col]   # Read a cell value from a Raster
            if elev >= 33.7 and elev != flow_accum.configs.nodata:
                path_length[row, col] = 1.0
            elif elev < 33.7 or elev == flow_accum.configs.nodata:
                path_length[row, col] = 0.0

    path = []
    for row in range(path_length.configs.rows):
        for col in range(path_length.configs.columns):
            path.append(path_length[row, col])

    path_sum = sum(path)
    return path_sum


def velocity_calculation(var_list):
    i = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                cut_and_fill[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                cut_and_fill[row, col] = var_list[i]
                i = i + 1

    # creat dem_pop_02
    dem_pop_02 = wbe.raster_calculator(expression="'dem' - 'cut_and _fill'", input_rasters=[dem, cut_and_fill])

    # path length calculation
    flow_accum_02 = wbe.d8_flow_accum(dem_pop_02, out_type='cells')
    slope = wbe.slope(dem_pop_02, units="percent")

    velocity = wbe.new_raster(flow_accum_02.configs)

    for row in range(flow_accum_02.configs.rows):
        for col in range(flow_accum_02.configs.columns):
            velo = flow_accum_02[row, col]
            if velo == flow_accum_02.configs.nodata:
                velocity[row, col] = slope.configs.nodata
            elif velo != flow_accum_02.configs.nodata:
                slope_factor = (slope[row, col] / 100) ** 0.5
                flow_factor = (flow_accum_02[row, col] * 100 * 0.000004215717) ** (2 / 3)
                velocity[row, col] = (slope_factor * flow_factor / 0.03) ** 0.6

    velocity_value = []
    for row in range(velocity.configs.rows):
        for col in range(velocity.configs.columns):
            velocity_value.append(velocity[row, col])

    max_velocity = max(velocity_value)
    return max_velocity

problem = MyProblem(n_grid)


# choose algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

algorithm = NSGA2(
    pop_size=20,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)


termination = get_termination("n_gen", 200)

from pymoo.optimize import minimize
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

X = res.X
F = res.F



### ----------------------------------------------- ###
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

# 3D Visualization
plot = Scatter(tight_layout=True)
plot.add(F, s=10)
plot.show()

# 2D Pairwise Scatter Plots
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=20, facecolors='none', edgecolors='blue')
plt.title("Flow path length (y) and total cost (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 1], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Max velocity (y) and flow path length (x)")
plt.grid()
plt.show()

plt.scatter(F[:, 0], F[:, 2], s=20, facecolors='none', edgecolors='blue')
plt.title("Max velocity (y) and total cost (x)")
plt.grid()
plt.show()


# save the solution result
result_df = pd.DataFrame(F)
result_df.to_csv('solution_output.csv', index=False)

# save the solution tif image
wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\08 Conference and activity\07 DLA Conference\AR_Sandbox_MOO\02_Solution_results'
for i in range(20):
    solution = res.X[i]
    solution_dem = wbe.new_raster(dem.configs)

    p = 0
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            if dem[row, col] == dem.configs.nodata:
                solution_dem[row, col] = dem.configs.nodata
            elif dem[row, col] != dem.configs.nodata:
                solution_dem[row, col] = solution[p]
                p = p + 1

    after_dem = dem - solution_dem
    filename = f'DEM_after_{i + 1}.tif'    #地形改动之后的结果
    wbe.write_raster(after_dem, file_name=filename, compress=True)

    filename_X = f'DEM_solution_{i + 1}.tif'   #地形自身的改动量
    wbe.write_raster(solution_dem, file_name=filename_X, compress=True)


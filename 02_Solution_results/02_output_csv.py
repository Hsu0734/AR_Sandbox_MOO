import whitebox_workflows as wbw
import pandas as pd
import numpy as np



wbe = wbw.WbEnvironment()
wbe.verbose = False
wbe.working_directory = r'D:\PhD career\08 Conference and activity\07 DLA Conference\AR_Sandbox_MOO\02_Solution_results'
# 定义DEM文件路径列表
dem_files = []

for i in range(20):
    filename = f'DEM_solution_{i + 1}.tif'
    dem_files.append(filename)

Elevation_point = []
# 遍历DEM文件
for i, dem_file in enumerate(dem_files):
    # 打开DEM文件
    dem = wbe.read_raster(f'DEM_solution_{i + 1}.tif')
    for row in range(dem.configs.rows):
        for col in range(dem.configs.columns):
            elev = dem[row, col]
            Elevation_point.append(elev)
    Elevation_point.append(np.nan)

result_df = pd.DataFrame(Elevation_point)
result_df.to_csv('elevation_point_output.csv', index=False)

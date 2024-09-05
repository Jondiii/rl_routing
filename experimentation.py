import subprocess
import pandas as pd

df_Experimentos = pd.read_csv('results_5Actions.csv', sep=';')
listaExperimentos = list(df_Experimentos['run_name'])

for experimento in listaExperimentos:

    command = [
        'python', 
        'main.py', 
        '--run_name', experimento, 
        '--dir_data', 'data/solomon_dataset/C1',  # si no encuentra el path en local probar con \\
        '--file_nodes', 'C101', 
        '--file_vehicles', 'vehicles/c1_vehicles',
        '--iterations', '100',
        '--timesteps', '102400'
    ]

    _ = subprocess.run(command, capture_output=True, text=True)

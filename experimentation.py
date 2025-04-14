import concurrent.futures
import subprocess
import pandas as pd
import os

df_experiments = pd.read_csv('experimentationResults.csv', sep=',')

experiments_dir = 'experiments'


def runExperiment(experiment):
        experimentName = experiment['ID']
        actionSpaceSize = experiment['Action Space Size']
        algorithm = experiment['Algorithm']
        timeWindows = True if experiment['Time Windows'] == 'TW' else False
        maxNodes = experiment['No. Nodes']

        if maxNodes < actionSpaceSize:
            return

        if timeWindows:
            nodeFile = 'C101'
        else:
            nodeFile = 'C101-NOTW'

        command = [
            'python', 
            'main.py', 
            '--run_name', experimentName, 
            '--dir_data', 'data/solomon_dataset/C1',  
            '--file_nodes', nodeFile, 
            '--file_vehicles', 'vehicles/c1_vehicles',
            '--iterations', '5',
            '--timesteps', '102400',
            '--save_model', 'no',
            '--save_logs', 'no',
            '--save_last_solution', 'no',
            '--action_space_size', str(actionSpaceSize),
            '--algo', algorithm,
            '--node_limit' , str(maxNodes),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True)
            result.check_returncode()  # Lanza un error si el código de salida no es 0
        except subprocess.CalledProcessError as e:
            print(f"Error en el experimento {experimentName}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")



if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        # TODO
        # - Hacer que se guarden los resultados en el csv de experimentationResults.csv
        
        future_to_experiment = {executor.submit(runExperiment, experiment) : experiment['ID'] for _,experiment in df_experiments.iterrows()}


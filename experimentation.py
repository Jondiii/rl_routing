import concurrent.futures
import subprocess
import pandas as pd
import os


experiments_dir = 'experiments'
experimentationResultsFile = 'experimentationResultsOnly5.csv'

df_experiments = pd.read_csv(experimentationResultsFile, sep=',')


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
            '--iterations', '1',#5
            '--timesteps', '2400',#102400
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

def update_csv(df, experiments_dir, experimentationResultsFile):
    for idx, row in df.iterrows():
        experimentName = row['ID']
        file_path = os.path.join(experiments_dir, f"{experimentName}.txt")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results = f.readline().strip().split(';')
                df.at[idx, 'Distance'] = float(results[0])
                df.at[idx, 'No. Vehicles'] = int(results[1])

    df.to_csv(experimentationResultsFile, index=False)

if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        # TODO
        # - Hacer que se guarden los resultados en el csv de experimentationResults.csv
        
        future_to_experiment = {executor.submit(runExperiment, experiment) : experiment['ID'] for _,experiment in df_experiments.iterrows()}

update_csv(df_experiments, experiments_dir, experimentationResultsFile)


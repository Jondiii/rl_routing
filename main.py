import argparse
from trainingManager import TrainingManager

#python main.py --run_name prueba --dir_data data

parser = argparse.ArgumentParser()
trainingManager = TrainingManager()

parser.add_argument("--run_name", help = "Name of the training run")
parser.add_argument("--dir_model", help = "Directory for the models", default="models")
parser.add_argument("--dir_log", help = "Directory for the logs", default="logs")
parser.add_argument("--dir_data", help = "Data directory")
parser.add_argument("--iterations", help = "Number of iterations to run the training for. A iteration is completed every n_timesteps.", default=10)
parser.add_argument("--timesteps", help = "Number of timesteps to run in each iteration.", default=20480)
parser.add_argument("--file_nodes", help = "File with node information", default=None)
parser.add_argument("--file_vehicles", help = "File with vehicle information", default=None)
parser.add_argument("--mode", help = "new_training or load_model", default='new_training')
parser.add_argument("--algo", help = "Algorithm to be used", default='PPO')
parser.add_argument("--render_mode", help = "Human", default=None)


# Read arguments from command line
args = parser.parse_args()

if args.dir_data is None:
    if (args.file_nodes is None) & (args.file_vehicles is None):
        raise ValueError("Node and vehicle information missing, please specify the data path with --dir_data or the file names with --file_nodes and --file_vehicles")

if args.mode == 'new_training':

    trainingManager.newTraining(
                    dataPath = args.dir_data,
                    algorithm = args.algo,
                    nodeFile = args.file_nodes,
                    vehicleFile = args.file_vehicles,
                    iterations = args.iterations,
                    timesteps = args.timesteps,
                    render_mode = args.render_mode
                )


elif args.mode == 'load_model':
    raise NotImplementedError("Functionality not yet implemented.")

else:
    raise ValueError("Choose between new_training or load_model modes with --mode")
import argparse
from trainingManager import TrainingManager

# SOLOMON benchmark TRAIN
# python main.py --run_name solomon_c1 --dir_data data\solomon_dataset\C1 --file_nodes C101 --file_vehicles vehicles\c1_vehicles     

# SOLOMON benchmark obtain solution
# C:\Users\jon.diaz\Desktop\codigo\rl_routing>python main.py --run_name solomon_c1_PPO --dir_data data\solomon_dataset\C1 --file_nodes C101 --file_vehicles vehicles\c1_vehicles --dir_model solomon_c1\models\ --iterations 1 --mode generate_Routes

parser = argparse.ArgumentParser()

parser.add_argument("--run_name", help = "Name of the training run. If loading a model, name of the model")
parser.add_argument("--dir_model", help = "Directory for the models", default="models")
parser.add_argument("--dir_log", help = "Directory for the logs", default="logs")
parser.add_argument("--dir_data", help = "Data directory")
parser.add_argument("--iterations", help = "Number of iterations to run the training for. A iteration is completed every n_timesteps.", default=10)
parser.add_argument("--timesteps", help = "Number of timesteps to run in each iteration.", default=20480)
parser.add_argument("--file_nodes", help = "File with node information", default=None)
parser.add_argument("--file_vehicles", help = "File with vehicle information", default=None)
parser.add_argument("--mode", help = "new_training or generate_Routes", default='new_training')
parser.add_argument("--algo", help = "Algorithm to be used", default='PPO')
parser.add_argument("--render_mode", help = "Human, optional", default=None)
parser.add_argument("--max_vehicles", help = "Maximum number of vehicles", default=None)
parser.add_argument("--name_model", help = "Name of the model", default=None)

# Read arguments from command line
args = parser.parse_args()

if args.dir_data is None:
    if (args.file_nodes is None) & (args.file_vehicles is None):
        raise ValueError("Node and vehicle information missing, please specify the data path with --dir_data or the file names with --file_nodes and --file_vehicles")

trainingManager = TrainingManager(run_name = args.run_name)


if args.mode == 'new_training':
    #python main.py --run_name prueba --dir_data data

    trainingManager.newTraining(
                    dataPath = args.dir_data,
                    algorithm = args.algo,
                    nodeFile = args.file_nodes,
                    vehicleFile = args.file_vehicles,
                    iterations = args.iterations,
                    timesteps = args.timesteps,
                    render_mode = args.render_mode,
                    max_vehicles = args.max_vehicles
                )


elif args.mode == 'generate_Routes':
    #python main.py --run_name prueba --dir_data data --dir_model prueba\models\ --iterations 1 --mode generate_Routes

    trainingManager.generateRoutes(
                    dataPath = args.dir_data,
                    algorithm = args.algo,
                    nodeFile = args.file_nodes,
                    vehicleFile = args.file_vehicles,
                    episodes = args.iterations,
                    render_mode = args.render_mode,
                    max_vehicles = args.max_vehicles,
                    dir_model = args.dir_model,
                    name_model = args.name_model,
    )

else:
    raise ValueError("Choose between new_training or generate_Routes modes with --mode")
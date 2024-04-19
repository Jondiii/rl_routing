import argparse
 

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("--name_train", help = "Name of the training run")
parser.add_argument("--dir_model", help = "Directory for the models", default="models")
parser.add_argument("--dir_log", help = "Directory for the logs", default="logs")
parser.add_argument("--n_iterations", help = "Number of iterations to run the training for. A iteration is completed every n_timesteps.")
parser.add_argument("--n_timesteps", help = "Number of timesteps to run in each iteration.", default=2048)
parser.add_argument("--n_vehicles", help = "Number of vehicles")
parser.add_argument("--n_nodes", help = "Number of nodes")
parser.add_argument("--max_n_vehicles", help = "Maxium number of vehicles available")
parser.add_argument("--max_n_nodes", help = "Maxium number of nodes available")


# Read arguments from command line
args = parser.parse_args()
 
if args.Output:
    print("Displaying Output as: % s" % args.Output)



from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO, A2C, DQN
from saveBestSolutionCallback import SaveBestSolutionCallback

import gymnasium as gym
import pandas as pd
import os
import time


class TrainingManager:
    
    def __init__(self, run_name, dir_model, dir_log, dir_experiments, save_logs, save_model, save_last_solution):
        
        self.run_name = run_name
        self.dir_model = os.path.join(self.run_name, dir_model)
        self.dir_log = os.path.join(self.run_name, dir_log)
        self.dir_experiments = dir_experiments

        self.save_logs = True if save_logs == 'yes' else False
        self.save_model = True if save_model == 'yes' else False
        self.save_last_solution = True if save_last_solution == 'yes' else False


        if self.save_model:
            if not os.path.exists(self.dir_model):
                os.makedirs(self.dir_model)

        if self.save_logs:
            if not os.path.exists(self.dir_log):
                os.makedirs(self.dir_log)


    def newTraining(self, 
                    dataPath,
                    algorithm,
                    nodeFile,
                    vehicleFile,
                    iterations,
                    timesteps,
                    render_mode,
                    max_vehicles,
                    action_space_size,
                    verbose,
                    nodeLimit,
                    ):
        
        self.iterations = int(iterations)
        self.timesteps = int(timesteps)

        self.run_name = self.run_name

        nodeLimit = int(nodeLimit) if nodeLimit else None

        if nodeFile:
            if vehicleFile: 
                self.env = gym.make('rl_routing:VRPEnv-v0',
                                    dataPath = dataPath,
                                    nodeFile = nodeFile,
                                    vehicleFile = vehicleFile,
                                    run_name=self.run_name,
                                    render_mode=render_mode,
                                    action_space_size=int(action_space_size),
                                    save_last_solution=self.save_last_solution,
                                    nodeLimit=nodeLimit,
                                    )
        else:
            self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode, action_space_size=int(action_space_size))

        if algorithm == 'PPO':
            self.model = PPO("MultiInputPolicy", self.env, verbose=verbose, tensorboard_log=self.dir_log if self.save_logs else None)

        elif algorithm == 'DQN':
            self.model = DQN("MultiInputPolicy", self.env, verbose=verbose, tensorboard_log=self.dir_log if self.save_logs else None)
        
        elif algorithm == 'A2C':
            self.model = A2C("MultiInputPolicy", self.env, verbose=verbose, tensorboard_log=self.dir_log if self.save_logs else None)
        
        else:
            raise ModuleNotFoundError("Reiforcement Learning algorithm not found")


        #stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=verbose)
        #eval_callback = EvalCallback(self.env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=verbose)
        save_best_solution_callback = SaveBestSolutionCallback(env = self.env, verbose=verbose, experiments_dir = self.dir_experiments)

        start_time = time.time()

        if self.save_model:
            for _ in range(1, int(self.iterations)+1):
                #self.model.learn(total_timesteps = self.timesteps, reset_num_timesteps = False, tb_log_name = self.run_name, callback = [eval_callback,save_best_solution_callback])
                self.model.learn(total_timesteps = self.timesteps, reset_num_timesteps = False, tb_log_name = self.run_name, callback = save_best_solution_callback)
                self.model.save(f"{self.dir_model}/{self.run_name}")

        else:
            for _ in range(1, int(self.iterations)+1):
                #self.model.learn(total_timesteps = self.timesteps, reset_num_timesteps = False, tb_log_name = self.run_name, callback = [eval_callback,save_best_solution_callback])
                self.model.learn(total_timesteps = self.timesteps, reset_num_timesteps = False, tb_log_name = self.run_name, callback = save_best_solution_callback)

        print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

        # Descomentar para guardar estos valores en un fichero
        #self.saveMetrics(self.dir_log+'/'+self.run_name+'_0', 'results_5Actions.csv', round((time.time() - start_time)/60, 2), ['rollout/ep_len_mean', 'rollout/ep_rew_mean'])

        self.env.close()





    def generateRoutes(self,
                       dataPath,
                       algorithm,
                       nodeFile,
                       vehicleFile,
                       dir_model,
                       episodes,
                       max_vehicles,
                       render_mode,
                       name_model,
                       size_action_space,
                       ):

        if name_model:
            model_path = f"{dir_model}/{name_model}"
        else:
            model_path = f"{dir_model}/{self.run_name}"


        if nodeFile:
            if vehicleFile:
                self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, nodeFile = nodeFile, vehicleFile = vehicleFile, run_name=self.run_name, render_mode=render_mode, n_visible_nodes=int(size_action_space))
        else:
            self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode, n_visible_nodes=int(size_action_space))

        if algorithm == 'PPO':
            self.model = PPO.load(model_path, self.env)

        elif algorithm == 'DQN':
            self.model = DQN.load(model_path, self.env)
        
        elif algorithm == 'A2C':
            self.model = A2C.load(model_path, self.env)
        
        else:
            raise ModuleNotFoundError("Reiforcement Learning algorithm not found")
        
        start_time = time.time()
        vec_env = self.model.get_env()

        for ep in range(int(episodes)):
            obs = vec_env.reset()
            dones = False
                
            while not dones:
                action, _ = self.model.predict(obs)

                obs, _, dones, _ = vec_env.step(action)

            vec_env.close()

            print("--- (%s/%s) %s minutos ---" % (ep+1, episodes, round((time.time() - start_time)/60, 2)))


    def saveMetrics(self, logdir, filePath, time, tags: list):
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()  
 
        metrics = {tag: [] for tag in tags}
        steps = []

        # Iterar sobre los eventos y almacenar las métricas
        for tag in tags:
            events = event_acc.Scalars(tag)
            for event in events:
                metrics[tag].append(event.value)
                steps.append(event.step)


        df = pd.read_csv(filePath, sep=';')

        df.loc[df['run_name'] == self.run_name, 'mean_ep_length'] = metrics['rollout/ep_len_mean'][-1]
        df.loc[df['run_name'] == self.run_name, 'mean_reward'] = metrics['rollout/ep_rew_mean'][-1]
        df.loc[df['run_name'] == self.run_name, 'training_time'] = round(time,3)

        df.to_csv(filePath, index=False, sep=';')


    """
    Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
    """

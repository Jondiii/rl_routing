from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO, A2C, DQN

import gymnasium as gym
import pandas as pd
import os
import time


class TrainingManager:
    
    def __init__(self, run_name = "run_name", dir_model = "models", dir_log = "logs"):
        self.run_name = run_name
        self.dir_model = os.path.join(self.run_name, dir_model)
        self.dir_log = os.path.join(self.run_name, dir_log)

        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)

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
                    ):
        
        self.iterations = int(iterations)
        self.timesteps = int(timesteps)
        
        self.run_name = self.run_name# + '_' + algorithm

        if nodeFile:
            if vehicleFile: 
                self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, nodeFile = nodeFile, vehicleFile = vehicleFile, run_name=self.run_name, render_mode=render_mode)
        else:
            self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode)

        if algorithm == 'PPO':
            self.model = PPO("MultiInputPolicy", self.env, verbose=1, tensorboard_log=self.dir_log)

        elif algorithm == 'DQN':
            self.model = DQN("MultiInputPolicy", self.env, verbose=1, tensorboard_log=self.dir_log)
        
        elif algorithm == 'A2C':
            self.model = A2C("MultiInputPolicy", self.env, verbose=1, tensorboard_log=self.dir_log)
        
        else:
            raise ModuleNotFoundError("Reiforcement Learning algorithm not found")

        start_time = time.time()

        for _ in range(1, int(self.iterations)+1):
            self.model.learn(total_timesteps = self.timesteps, reset_num_timesteps = False, tb_log_name = self.run_name)
            self.model.save(f"{self.dir_model}/{self.run_name}")

        print("--- %s minutos ---" % round((time.time() - start_time)/60, 2))

        self.saveMetrics(self.dir_log+'/'+self.run_name+'_0', 'results_infiniteActions.csv', round((time.time() - start_time)/60, 2), ['rollout/ep_len_mean', 'rollout/ep_rew_mean'])

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
                       name_model
                       ):
        
        if name_model:
            model_path = f"{dir_model}/{name_model}"
        else:
            model_path = f"{dir_model}/{self.run_name}"


        if nodeFile:
            if vehicleFile:
                self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, nodeFile = nodeFile, vehicleFile = vehicleFile, run_name=self.run_name, render_mode=render_mode)
        else:
            self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode)

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

        df = pd.read_csv(filePath, sep=';')

        try:
            for tag in tags:
                events = event_acc.Scalars(tag)
                for event in events:
                    metrics[tag].append(event.value)
                    steps.append(event.step)

            df.loc[df['run_name'] == self.run_name, 'mean_ep_length'] = metrics['rollout/ep_len_mean'][-1]
            df.loc[df['run_name'] == self.run_name, 'mean_reward'] = metrics['rollout/ep_rew_mean'][-1]
            df.loc[df['run_name'] == self.run_name, 'training_time'] = "--- %s minutos ---" %time

            df.to_csv(filePath, index=False, sep=';')

        except Exception as e:
            df.loc[df['run_name'] == self.run_name, 'mean_ep_length'] = -999999999
            df.loc[df['run_name'] == self.run_name, 'mean_reward'] = -999999999

            df.to_csv(filePath, index=False, sep=';')



    """
    Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
    """

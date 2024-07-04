from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, DummyVecEnv, VecEnvWrapper

from rl_routing.envs.vrpEnv import VRPEnv

import gymnasium as gym
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
        
        self.iterations = iterations
        self.timesteps = timesteps
        
        self.run_name = self.run_name + '_' + algorithm

        self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode)


        if nodeFile:
            if vehicleFile: # Sólo cambia que aquí no se usas max_vehicles y en el anterior sí
                self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, nodeFile = nodeFile, vehicleFile = vehicleFile, run_name=self.run_name, render_mode=render_mode)

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

        self.env.close()


    def generateRoutes(self,
                       dataPath,
                       algorithm,
                       nodeFile,
                       vehicleFile,
                       dir_model,
                       episodes,
                       max_vehicles,
                       render_mode
                       ):
        
        model_path = f"{dir_model}/{self.run_name}"

        self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, max_vehicles = max_vehicles, run_name=self.run_name, render_mode=render_mode)

        if nodeFile:
            if vehicleFile:
                self.env = gym.make('rl_routing:VRPEnv-v0', dataPath = dataPath, nodeFile = nodeFile, vehicleFile = vehicleFile, run_name=self.run_name, render_mode=render_mode)

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




    """
    Nota: usar el comando python crearYEntrenar.py >> log.txt 2>> errLog.txt para redirigir las salidas de prints/logs y errores.   
    """

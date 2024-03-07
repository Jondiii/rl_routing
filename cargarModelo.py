from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor, DummyVecEnv, VecEnvWrapper

from rl_routing.envs.vrpEnv import VRPEnv

import gymnasium as gym
import os
import time

"""
Definimos primero dónde buscar el modelo ya entrenado.
"""

model_name = "temp/2048.zip"
models_dir = "temp" # Sin el -1 de las acciones, no funciona ni tan mal, pero tarda la vida. 
model_path = f"{models_dir}/{model_name}"

"""
INICIALIZACIÓN DE ENTORNO Y AGENTE
"""
nVehiculos = 7
nNodos = 20

env = gym.make('rl_routing:VRPEnv-v0',  nVehiculos = nVehiculos, nNodos = nNodos, maxNodeCapacity = 4, sameMaxNodeVehicles=True, render_mode='human')

env.reset()

model = PPO.load(model_path, env)
vec_env = model.get_env()

# Indicamos el número de episodios (a más episodios más soluciones obtendremos)
episodes = 1

start_time = time.time()

"""
GENERACIÓN DE RUTAS
"""
for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)

        obs, reward, done, info = vec_env.step(action)

    vec_env.render('human') # Guarda un report y los grafos en la ruta especificada.

    vec_env.close()
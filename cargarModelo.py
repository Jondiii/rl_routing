from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
import time

"""
Definimos primero dónde buscar el modelo ya entrenado.
"""

model_name = "baseAlg/102400.zip"
models_dir = "models" # Sin el -1 de las acciones, no funciona ni tan mal, pero tarda la vida. 
model_path = f"{models_dir}/{model_name}"

"""
INICIALIZACIÓN DE ENTORNO Y AGENTE
"""
nVehiculos = 7
nNodos = 20

env = gym.make('rl_routing:VRPEnv-v0',  nVehiculos = 7, nNodos = 20, maxNumVehiculos = 7, maxNumNodos = 20, maxNodeCapacity = 4, sameMaxNodeVehicles=False, render_mode='human', dataPath = 'data/')
#env.readEnvFromFile(nVehiculos = 5, nNodos = 15, maxVehicles = 7, maxNodos = 20, dataPath = 'data/')
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
        vec_env.render()
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()
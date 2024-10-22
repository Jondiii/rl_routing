import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from rl_routing.envs.vrpEnv import VRPEnv

"""
Gymnasium contiene una variedad de entornos listos para usar, pero también permite crear entornos propios.
El método env_checker sirve para comprobar que los entornos creados por los usuarios sean consistentes en cuanto
al espacio de acciones y de observaciones, mirando que los espacios declarados coinciden con las observaciones
que devuelve el entorno en cada iteración.

Que un entorno pase el checker no quiere decir que esté libre de errores.
"""

#with open('envs.txt', 'w') as f:
#    f.write(str(gym.envs.registry.keys()))

# Se crea un entorno de prueba.
#env = gym.make('rl_routing:VRPEnv-v0', nVehiculos = 5, nNodos = 20, sameMaxNodeVehicles=True)

env = gym.make('rl_routing:VRPEnv-v0', dataPath = 'data', max_vehicles = 10, run_name='lala', render_mode=None)

# Se comprueba que el entorno sea consistente.
sb3_check_env(env)

#check_env(env)


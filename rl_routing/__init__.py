from gymnasium.envs.registration import register

register(
     id="VRPEnv-v0", 
     entry_point="rl_routing.envs.vrpEnv:VRPEnv",
) # If id = "rl_routing:VRPEnv-v0", then gym.make rasises the error: gymnasium.error.NameNotFound: Environment `VRPEnv` doesn't exist.

register(
     id="VRPEnvVec-v0", 
     entry_point="rl_routing.envs.vrpEnv:VRPEnvVec",
)
from gymnasium.envs.registration import register

register(
     id="rl_routing:VRPEnv-v0",
     entry_point="rl_routing.envs.vrpEnv:VRPEnv",
)
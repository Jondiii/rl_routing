from stable_baselines3.common.callbacks import BaseCallback
import os

class SaveBestSolutionCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, env = None, verbose: int = 0, experiments_dir: str = "experiments") -> None:
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.env = env

        self.experiments_dir = experiments_dir
        
        self.run_name = self.env.get_wrapper_attr('run_name')
        with open(os.path.join(self.experiments_dir, "{}.txt".format(self.run_name)), "w") as f:
            f.write("500000;5000")
            f.close()


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.env.get_wrapper_attr('terminated'):
            new_solution = self.env.get_wrapper_attr('grafoCompletado')

            if new_solution is None:
                return True
            
            new_totalDistance = new_solution.getTotalDistance()   
            new_nVehicles = len(new_solution.rutas)

            old_totalDistance = 0.0
            old_nVehicles = 0


            with open(os.path.join(self.experiments_dir, "{}.txt".format(self.run_name)), "r") as f:
                firstLine = f.readline()
                firstLineItems = firstLine.split(";")

                old_totalDistance = float(firstLineItems[0])
                old_nVehicles = int(firstLineItems[1])
                f.close()

            with open(os.path.join(self.experiments_dir, "{}.txt".format(self.run_name)), "w") as f:
                if new_totalDistance < old_totalDistance:
                    f.write("{};{}\n".format(new_totalDistance, new_nVehicles))
                else:
                    f.write("{};{}\n".format(old_totalDistance, old_nVehicles))

                f.close()

        return True

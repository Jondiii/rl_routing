from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from sklearn.metrics.pairwise import euclidean_distances
import gymnasium as gym
import numpy as np
from utils.solution import Solution
from utils.dataGenerator import DataGenerator
from utils.dataReader import DataReader
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
import copy
import os
from datetime import date


class VRPEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps' : 5}
    run_name = None
    seed = None

    decayingStart = None
    
    grafoCompletado = None
    fig = None

    nNodos = 0
    nVehiculos = 0

    closestNodes = None

    n_coordenadas = None
    n_originalDemands = None
    n_demands = None
    n_maxNodeCapacity = None
    n_twMin = None
    n_twMax = None

    v_load = None
    v_maxCapacity = None

    
    def __init__(self, dataPath = None, max_vehicles = None, nodeFile = 'nodes', vehicleFile = 'vehicles', maxSteps = np.nan,
                seed = None, singlePlot = False, run_name = None, graphSavePath = None, render_mode = None,
                action_space_size = 5):

        super(VRPEnv, self).__init__()

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert action_space_size >= 1

        self.render_mode = render_mode

        self.singlePlot = singlePlot
        self.maxSteps = maxSteps
        self.currTotalSteps = 0
        self.currEpisodeSteps = 0
        self.run_name = run_name
        self.graphSavePath = graphSavePath
        self.action_space_size = action_space_size # Contando el depot pero no las 2 acciones extra

        self.isDoneFunction = self.isDone

        self.readEnvFromFile(dataPath, nodeFile, vehicleFile)

        # Cálculo de matrices de distancia
        self.createMatrixes()

        # Creamos el espacio de acciones y el espacio de observaciones
        self.createSpaces()


        if not max_vehicles:
            self.max_vehicles = len(self.vehicleInfo)
        else:
            self.max_vehicles = max_vehicles



    # Método que creará un entorno a partir de lo que se haya almacenado en los ficheros.
    def readEnvFromFile(self, dataPath, nodeFile, vehicleFile):
        self.dataReader  = DataReader(dataPath, nodeFile, vehicleFile)

        self.nodeInfo = pd.DataFrame(self.dataReader.loadNodeData())
        self.nodeInfo['is_visited'] = 0
        self.nodeInfo = self.nodeInfo.reset_index()
        self.vehicleInfo = pd.DataFrame(self.dataReader.loadVehicleData())

        self.nNodos = len(self.nodeInfo)
        self.nVehiculos = len(self.vehicleInfo)

        # Características de los nodos
        self.n_coordenadas = np.array([self.nodeInfo["coordenadas_X"], self.nodeInfo["coordenadas_Y"]]).T
        self.n_maxNodeCapacity = max(self.nodeInfo['demandas'])

        # Características de los vehículos
        self.v_load = self.vehicleInfo["maxCapacity"].to_numpy()
        self.v_maxCapacity = self.v_load.max()

        # Ventanas de tiempo y service time
        self.n_twMin = self.nodeInfo["minTW"].to_numpy()
        self.n_twMax = self.nodeInfo["maxTW"].to_numpy()
        self.n_serviceTime = self.nodeInfo["service_time"].to_numpy()


    # Método que creará el espacio de acciones y el de observaciones.
    def createSpaces(self):
        self.action_space = Discrete(self.action_space_size) # Nodos (sumando depot) + 2 acciones extra

        self.observation_space = Dict({
            "v_curr_position" : Discrete(self.nNodos), # Se almacena la posición actual
            "v_load" : Discrete(self.v_maxCapacity + 1), # SOLO se pueden usar enteros
            "n_demands" : MultiDiscrete(np.zeros(shape=self.action_space_size) + self.n_maxNodeCapacity+1, dtype=np.int64),
            "n_distances" : Box(low = 0, high = float('inf'), shape = (self.action_space_size,), dtype=np.float64),
        })

    # Método encargado de ejecutar las acciones seleccionadas por el agente.
    def step(self, action):
        self.currTotalSteps += 1 
        self.currEpisodeSteps += 1
        
        # Calcular el nodo a visitar
        node = self.closestNodes['index'][action]
        distancia = self.getDistanceTime(action)

        if not self.checkAction(node, distancia):
            return self._get_obs(), -1, False, self.isTruncated(), dict()

        # Añadimos la demanda del nodo a la carga del vehículo
        self.v_load-= self.n_demands[action]

        self.visitEdge(node, distancia)
        
        self.v_posicionActual = int(node)

        if node == 0:
            self.v_load = self.v_maxCapacity
            self.solution.nuevaRuta()

        self.closestNodes = self.getClosestNodes(self.action_space_size)

        self.n_demands = np.array(self.closestNodes['demandas'], dtype=np.int64)
        self.n_distances = np.array(self.closestNodes['distances'], dtype=np.float64)

        truncated = self.isTruncated()
        terminated = self.isDoneFunction()

        return self._get_obs(), self.getReward(distancia, node, terminated, truncated), terminated, truncated, self._get_info()


    # Método para resetear el entorno. Se hace al principio del entrenamiento y tras cada episodio.
    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        # Marcamos los nodos como no visitados
        self.nodeInfo['is_visited'] = 0

        # Reiniciamos el vehículo
        self.v_posicionActual = 0
        self.v_load = self.v_maxCapacity
        self.v_ordenVisitas = [0]

        self.currEpisodeSteps = 0

        self.closestNodes = self.getClosestNodes(self.action_space_size) # Todo se tiene que pillar de aquí.

        self.n_demands = np.array(self.closestNodes['demandas'], dtype=np.int64)
        self.n_distances = np.array(self.closestNodes['distances'], dtype=np.float64)


        # Creamos un conjunto de rutas nuevo
        self.solution = Solution(self.nNodos,  self.n_coordenadas)
               
        return self._get_obs(), self._get_info()


    # Método que obtiene las observaciones del entorno
    def _get_obs(self): # TODO the obs returned by the `step()` method should be an int or np.int64, actual type: <class 'numpy.int32'>
        obs = dict()
        obs["v_curr_position"] = self.v_posicionActual
        obs["v_load"] = self.v_load
        obs["n_demands"] = self.n_demands 
        obs["n_distances"] = self.n_distances

        return obs

    def _get_info(self, text = None):
        info = {'info' : str(text)}
        return info

    # Truncated se usa para indicar si el episodio ha finalizado porque se ha cumplido una condición fuera del
    # objetivo final del episodio, como haber alcanzado cierto límite de tiempo.
    def isTruncated(self):
        if self.maxSteps is None:
            return False

        return True if self.maxSteps <= self.currEpisodeSteps else False


    def isDone(self):
        """
        Método que comprueba si un episodio ha finalizado. Este finalizará si una de las dos se cumple:
        - Todos los nodos se han visitado
        - Se han empleado todos los vehículos disponibles
        """
        if np.all(self.nodeInfo['is_visited'] == 1):
            self.visitEdge(0, self.getDistanceTime(0))
            self.grafoCompletado = copy.deepcopy(self.solution) # Guardamos siempre el último conjunto de rutas completas, para poder dibujarlas al finalizar el entrenamiento.
            return True
        
        if len(self.solution.rutas) > self.max_vehicles:
            return True

        return False
    

    def checkAction(self, node, distancia):
        """
        Método que comprueba la validez de una acción. La acción será incorrecta si:
        - Se visita un nodo ya visitado
        - El vehículo no se mueve.
        - Se visita un nodo con una demanda superior a la que puede llevar el vehículo.
        - El vehículo llega al nodo antes del inicio de la ventana de tiempo o después del cierre de esta.
        """
        if self.v_posicionActual == node:
            return False
        
        if self.v_load - self.nodeInfo.loc[node, 'demandas'] < 0:
            return False

        #if self.n_twMin[node] > (self.solution.rutas[-1].travelDistance + distancia + self.n_serviceTime[node]):# el tiempo de ruta se reinicia con cada ruta. hay que hacer que se tenga un tiempo global y que sea eso lo que se comprueba
        #    print("MIN: {} - {}".format(self.n_twMin[node], self.solution.rutas[-1].travelDistance + distancia + self.n_serviceTime[node]))
        #    return False

        #if self.n_twMax[node] < (self.solution.rutas[-1].travelDistance + distancia + self.n_serviceTime[node]):
        #    print("MAX: {} - {}".format(self.n_twMax[node], self.solution.rutas[-1].travelDistance + distancia + self.n_serviceTime[node]))
        #    return False

        return True
    

    def getClosestNodes(self, lastNode):
        # Obtener la información del nodo en la posición 0 (depot)
        depot_info = self.nodeInfo.loc[0]

        # Distancias desde el nodo en v_posicionActual a todos los otros nodos
        distances = self.distanceMatrix[self.v_posicionActual]

        # Filtrar los nodos no visitados y excluir el nodo actual y el depot
        unvisited_nodes = self.nodeInfo[
            (self.nodeInfo['is_visited'] == 0) & 
            (self.nodeInfo.index != self.v_posicionActual) & 
            (self.nodeInfo.index != 0)
        ]

        # Obtener índices nodos no visitados
        unvisited_indices = unvisited_nodes.index

        # Obtener distancias de nodos no visitados
        filtered_distances = distances[unvisited_indices]
        filtered_distances_indexes = np.argsort(filtered_distances)

        if len(unvisited_indices) < lastNode:
            closest_distances_indexes = filtered_distances_indexes[:self.action_space_size -1]

            closest_indices = unvisited_indices[closest_distances_indexes]

            # Obtener distancias e info nodos no visitados más cercanos
            closest_nodes_info = self.nodeInfo.loc[closest_indices]
            closest_distances = filtered_distances[closest_distances_indexes]


        else:
            closest_distances_indexes = filtered_distances_indexes[lastNode-self.action_space_size:lastNode - 1]
            closest_indices = unvisited_indices[closest_distances_indexes]
            closest_nodes_info = self.nodeInfo.loc[closest_indices]

            # Obtener distancias nodos no visitados más cercanos
            closest_distances = filtered_distances[closest_distances_indexes]

        # Crear el diccionario con la información del depot y los nodos más cercanos
        result = {}
        for column in self.nodeInfo.columns:
            if column in ['coordenadas_X', 'coordenadas_Y']:
                values = [depot_info[column]] + closest_nodes_info[column].tolist()
            else:
                values = [int(depot_info[column])] + [int(x) for x in closest_nodes_info[column].tolist()]

            # Rellenar con copias del depot si hay menos de 4 nodos no visitados
            while len(values) < (self.action_space_size):
                values.append(depot_info[column] if column in ['coordenadas_X', 'coordenadas_Y'] else int(depot_info[column]))

            result[column] = np.array(values)

        # Añadir las distancias al diccionario result
        distances = [0] + closest_distances.tolist()
        while len(distances) < (self.action_space_size):
            distances.append(0)  # La distancia al depot es 0
        result['distances'] = np.array(distances)

        return result


    def getDistanceTime(self, action):
        distancia = self.n_distances[action]

        return distancia


    def visitEdge(self, node, distancia):
        """
        Marca el nodo correspondiente como visitado y se actualiza el Grafo.
        """

        self.nodeInfo.loc[node, 'is_visited'] = 1
        
        self.solution.visitEdge(self.v_posicionActual, node, distancia, self.n_serviceTime[node])




    def setIncreasingIsDone(self, totalSteps, minNumVisited = 0.5, increaseStart = 0.5, increaseRate = 0.1, everyNtimesteps = 0.1):
        """
        Método que añade los parámetros iniciales del método increasingIsDone.
        totalSteps: número total de pasos que durará el entrenamiento.
        minNumVisited: porcentaje mínimo de nodos que debe ser visitado al inicio del entrenamiento.
        increaseStart: porcentaje de pasos que deben haber pasado para que el porcentaje mínimo de nodos comience a aumentar.
        increaseRate: cuánto aumentará el porcentaje mínimo de nodos a visitar.
        everyNtimesteps: cada cuánto aumentará el porcentaje mínimo de nodos a visitar.
        """
        self.totalSteps = totalSteps
        self.increaseStart = increaseStart
        self.increaseRate = increaseRate
        self.everyNTimesteps = totalSteps * everyNtimesteps

        self.minimumVisited = minNumVisited

        self.isDoneFunction = self.increasingIsDone


    # Método que sustituye a isDone. Comenzará permitiendo no visitar todos los nodos, y a medida que el entrenamiento avance,
    # el porcentaje mínimo de nodos a visitar irá en aumento.
    def increasingIsDone(self):
        # Se calcula el porcentaje mínimo a visitar
        if self.currTotalSteps / self.totalSteps >= self.increaseStart:
            if self.currTotalSteps % self.everyNTimesteps == 0: 
                self.minimumVisited += self.increaseRate
                
                # Si se pasa de 100%, poner a 100%
                if self.minimumVisited >= 1:
                    self.minimumVisited = 1

   
        # Si no es multitrip, se tendrá que comprobar que todos han regresado al depot y que cumplen el porcentaje mínimo.
        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.n_visited[:self.nNodos] == 1) / self.nNodos
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.solution)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                return True
        
        # Si no están todos en el depot, se marca el depósito como no visitado para que el/los vehículos que queden puedan regresar.
        else:
            porcentajeVisitados = np.count_nonzero(self.n_visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.n_visited[0] = 0

        return False



    def setDecayingIsDone(self, totalSteps, decayingStart = 0.5, decayingRate = 0.1, everyNtimesteps = 0.05):
        """
        Método que añade los parámetros iniciales del método decayingIsDone.
        totalSteps: número total de pasos que durará el entrenamiento.
        decayingStart: porcentaje de pasos que deben haber pasado para que el porcentaje mínimo de nodos comience a disminuir.
        decayingRate: cuánto disminuirá el porcentaje mínimo de nodos a visitar.
        everyNtimesteps: cada cuánto disminuirá el porcentaje mínimo de nodos a visitar.
        """
        self.totalSteps = totalSteps
        self.decayingStart = decayingStart
        self.decayingRate = decayingRate
        self.everyNTimesteps = totalSteps * everyNtimesteps

        self.minimumVisited = 1

        self.isDoneFunction = self.decayingIsDone

    # Método que sustituye a isDone. Comenzará obligando a visitar todos los nodos, pero tras cierta cantidad de pasos
    # solo será necesario visitar una fracción de estos para poder dar por finalizar el episodio.
    def decayingIsDone(self):
        # Mientras no llevemos más de los pasos indicados en decayingStart, el isDone funciona de manera normal. 
        # Después de pasarlo, aplicaremos la reducción de porcentaje de nodos a visitar.
        if self.minimumVisited == 1:
            if self.currTotalSteps / self.totalSteps >= self.decayingStart: # Si se pasa de más de decayingStart%, se visitan menos
                self.minimumVisited -= self.decayingRate

            return self.isDone()
        
        # Vamos disminuyendo el mínimo a visitar cada everyNTimesteps, en una proporción de decayingRate
        if self.currTotalSteps % self.everyNTimesteps == 0:
            self.minimumVisited -= self.decayingRate
       
        # Si no es multitrip, solo comprobaremos si ha finalizado el episodio si los vehículos están en el depot
        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.n_visited[:self.nNodos] == 1) / self.nNodos
                        
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.solution)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                return True
        
        # Si no lo están, marcamos el depot como no visitado 
        else:
            porcentajeVisitados = np.count_nonzero(self.n_visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.n_visited[0] = 0

     
        return False
    
        
    # Método que calcula la recompensa a dar al agente.
    def getReward(self, distancia, node, terminated, truncated):
        if truncated:
            return -100
        
        if terminated:
            reward = self.getReward(distancia, node, False, False)

            nodesNotVisited = sum(self.nodeInfo['is_visited'] == 0)

            return reward  - nodesNotVisited * 10

        if distancia == 0:
            return -1

        # Cuando el vehículo termina su ruta recibe una recompensa inversamente proporcional a lo lleno que vaya el vehículo,
        # a más llenado más recompensa. Por defecto v_loads está a 100 (capacidad máxima), y se le va restando según se recogen pedidos.
        if node % self.nNodos == 0:
            if self.v_load != 0:
                return round(1/abs(self.v_load), 2) 
            return 1

        # La recompensa será inversamente proporcional a la distancia recorrida, a mayor distancia, menor recompensa
        return round(1/abs(distancia), 2)




    # Creamos las matrices de distancias y de tiempo. # TODO igual hacerlo con valhalla si es un caso real
    # Estas matrices tendrán ya calculados la distancia y el tiempo que hay entre cada par de nodos, de manera que en el resto del código,
    # para saber estos datos, bastará con consultar las matrices, ahorrando cálculos.
    def createMatrixes(self):
        self.distanceMatrix = euclidean_distances(X = np.array(self.n_coordenadas))

    # Se crean y añaden ventanas del tiempo al problema en función de los valores especificados.
    def crearTW(self, n_twMin, n_twMax):
        if n_twMin is None:
            self.n_twMin = np.zeros(shape=self.nNodos)
        else:
            self.n_twMin = np.zeros(shape=self.nNodos) + n_twMin

        if n_twMax is None:
            self.n_twMax = np.zeros(shape=self.nNodos) + 100000 # No deja poner inf, tensorflow lo interpreta como nan
        else:
            self.n_twMax = np.zeros(shape=self.nNodos) + n_twMax


    # Genera casos de forma semi-aleatoria haciendo uso de la clase dataGenerator
    def generateRandomData(self):
        self.dataGen = DataGenerator(self.nNodos, self.nVehiculos, seed = self.seed)

        self.dataGen.addNodeInfo(self.n_maxNodeCapacity, self.n_twMin, self.n_twMin)
        self.dataGen.generateNodeInfo()
        self.dataGen.addVehicleInfo(self.v_maxCapacity, self.v_speeds)
        self.dataGen.generateVehicleInfo()
        self.dataGen.saveData()

        self.loadData()

    # Cargamos los datos generados
    def loadData(self):
        # Características de los nodos
        self.n_coordenadas = np.array([self.dataGen.nodeInfo["coordenadas_X"], self.dataGen.nodeInfo["coordenadas_Y"]]).T
        self.n_originalDemands = self.dataGen.nodeInfo["demandas"].to_numpy()
        self.n_demands = copy.deepcopy(self.n_originalDemands) ## DEJAR
        self.n_maxNodeCapacity = self.dataGen.nodeInfo["maxDemand"][0] # El atributo maxDemand debería ser igual en todos los nodos del fichero

        # Características de los vehículos
        self.v_load = self.dataGen.vehicleInfo["maxCapacity"].to_numpy()
        self.v_maxCapacity = self.v_load.max()
        self.v_speeds = self.dataGen.vehicleInfo["speed"].to_numpy()

        # Ventanas de tiempo
        self.n_twMin = self.dataGen.nodeInfo["minTW"].to_numpy()
        self.n_twMax = self.dataGen.nodeInfo["maxTW"].to_numpy()

    #TODO creo que esto no se usa
    def renderRoutes(self, dir = 'default'):      
        if self.grafoCompletado == None:
            return
        
        # Llama a un método de guardado o a otro dependiendo de si se quieren todas las rutas en un mismo plot o no
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot(dir)

        else:
            self.grafoCompletado.guardarGrafoSolucion(dir)

        self.grafoCompletado.guardarTextoSolucion(dir)

        #self.crearReport(self.ordenVisitasCompletas, directorio = dir)


    # Crea y guarda una imagen y un report el último conjunto de grafos completado 
    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.num_columns = min(self.nVehiculos, 4)
            self.num_rows = np.ceil(self.nVehiculos / self.num_columns).astype(int)

        plt.clf()

        for n, idGrafo in enumerate(range(len(self.solution.rutas))):
            ax = plt.subplot(self.num_rows, self.num_columns, n + 1)

            self.solution.rutas[idGrafo].dibujarGrafo(ax = ax)

        plt.pause(0.2) # PROBAR con esto: https://graphviz.readthedocs.io/en/stable/manual.html


    def close(self):
        super().close()
        
        if self.graphSavePath:
            self.generateReport(self.graphSavePath)
        else:
            self.generateReport(self.run_name)


    # Guarda el conjunto actual de grafos, independientemente de si están completos o no
    def generateReport(self, dir):
        if self.grafoCompletado == None:
            print("No se han podido generar rutas.")
            return


        # Llama a un método de guardado o a otro dependiendo de si se quieren todas las rutas en un mismo plot o no
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot(directorio = dir)

        else:
            self.grafoCompletado.guardarGrafoSolucion(directorio = dir)

        self.grafoCompletado.guardarTextoSolucion(dir)

from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from sklearn.metrics.pairwise import haversine_distances
import gymnasium as gym
import numpy as np
from utils.routes import Routes
from utils.dataGenerator import DataGenerator
from utils.dataReader import DataReader
import networkx as nx
import matplotlib.pyplot as plt
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
    travelTime = None


    n_coordenadas = None
    n_originalDemands = None
    n_demands = None
    n_maxNodeCapacity = None
    n_twMin = None
    n_twMax = None
    
    v_loads = None
    v_maxCapacity = None
    v_speeds = None



    
    def __init__(self, dataPath = None, nodeFile = 'nodes', vehicleFile = 'vehicles', maxSteps = None,
                seed = None,  multiTrip = False, singlePlot = False, 
                run_name = None, graphSavePath = None, render_mode = None):

        super(VRPEnv, self).__init__()

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        self.multiTrip = multiTrip
        self.singlePlot = singlePlot
        self.maxSteps = maxSteps
        self.currTotalSteps = 0
        self.currEpisodeSteps = 0
        self.run_name = run_name
        self.graphSavePath = graphSavePath

        self.travelTime = np.zeros(shape=(self.nVehiculos,1))

        self.isDoneFunction = self.isDone

        self.readEnvFromFile(dataPath, nodeFile, vehicleFile)

        # Cálculo de matrices de distancia
        self.createMatrixes()

        # Creamos el espacio de acciones y el espacio de observaciones
        self.createSpaces()


    # Método que creará un entorno a partir de lo que se haya almacenado en los ficheros.
    def readEnvFromFile(self, dataPath, nodeFile, vehicleFile):
        self.dataReader  = DataReader(dataPath, nodeFile, vehicleFile)

        nodeInfo = self.dataReader.loadNodeData()
        vehicleInfo = self.dataReader.loadVehicleData()

        self.nNodos = len(nodeInfo["coordenadas_X"])
        self.nVehiculos = len(vehicleInfo["maxCapacity"])

        # Características de los nodos
        self.n_coordenadas = np.array([nodeInfo["coordenadas_X"], nodeInfo["coordenadas_Y"]]).T
        self.n_originalDemands = nodeInfo["demandas"].to_numpy()
        self.n_demands = copy.deepcopy(self.n_originalDemands)
        self.n_maxNodeCapacity = nodeInfo["maxDemand"][0] # El atributo maxDemand debería ser igual en todos los nodos del fichero

        # Características de los vehículos
        self.v_loads = vehicleInfo["maxCapacity"].to_numpy()
        self.v_maxCapacity = self.v_loads.max()
        self.v_speeds = vehicleInfo["speed"].to_numpy()

        # Ventanas de tiempo
        self.n_twMin = nodeInfo["minTW"].to_numpy()
        self.n_twMax = nodeInfo["maxTW"].to_numpy()


    # Método que creará el espacio de acciones y el de observaciones.
    def createSpaces(self):
        self.action_space = Discrete(self.nNodos * self.nVehiculos -1)

        self.observation_space = Dict({
            "n_visited" :  MultiDiscrete(np.zeros(shape=self.nNodos) + 2),
            "v_curr_position" : MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.nNodos),
            "v_loads" : MultiDiscrete(np.zeros(shape=self.nVehiculos) + self.v_maxCapacity + 1), # SOLO se pueden usar enteros
            "n_demands" : MultiDiscrete(np.zeros(shape=self.nNodos) + self.n_maxNodeCapacity * 5),
            "n_distances" : Box(low = 0, high = float('inf'), shape = (self.nVehiculos * self.nNodos,), dtype=float),
        })

    # Método encargado de ejecutar las acciones seleccionadas por el agente.
    def step(self, action):
        self.currTotalSteps += 1 
        self.currEpisodeSteps += 1
        
        # Comprobamos que la acción sea sobre un nodo que forme parte del problema (relevante cuando nNodos != nMaxNodos)
        if action >= self.nNodos * self.nVehiculos:
            return self._get_obs(), 0, False, self.isTruncated(), dict(info = "Acción rechazada por actuar sobre un nodo no disponible.", accion = action, nNodos = self.nNodos)

        vehiculo = action // self.nNodos

        node = action % self.nNodos

        if not self.checkAction(node, vehiculo):
            return self._get_obs(), 0, False, self.isTruncated(), dict()

        # Eliminar el lugar que se acaba de visitar de las posibles acciones
        self.visited[node] = 1

        # Si se permite el multiTrip entonces un vehículo podrá pasar por el depot para vaciar su carga y continuar visitando nodos.
        if self.multiTrip:
            if node == 0: # Si la acción consiste en volver al depot...
                self.v_loads[vehiculo] = self.v_maxCapacity
                self.visited[node] = 0 # Si lo que se ha visitado es el depot, no lo marcamos como visitado 

        self.v_loads[vehiculo] -= self.n_demands[node] # Añadimos la demanda del nodo a la carga del vehículo

        distancia, tiempo = self.rutas.visitEdge(vehiculo, self.v_posicionActual[vehiculo], node)

        self.n_demands[node] = 0
        
        reward = self.getReward(distancia, node, vehiculo)

        self.v_posicionActual[vehiculo] = node

        self.v_ordenVisitas[vehiculo].append(node)

        self.n_distances[vehiculo] = self.distanceMatrix[node]

        self.travelTime[vehiculo] += tiempo

        terminated = self.isDoneFunction()

        
        return self._get_obs(), reward, terminated, self.isTruncated(), self._get_info()


    # Método para resetear el entorno. Se hace al principio del entrenamiento y tras cada episodio.
    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.visited = np.zeros(shape=(self.nNodos), dtype=np.int64) # Marcamos todo como no visitado.

        # Si nNodos != nMaxNodos, se deben marcar los nodos de sobra como visitados, para que no sean válidos en ningún caso.
        self.visited = np.pad(self.visited, (0,self.nNodos - self.nNodos), 'constant', constant_values = 1)

        if self.multiTrip:
            self.visited[0] = 0
        else:
            self.visited[0] = 1 # El depot comienza como visitado.

        self.v_posicionActual = np.zeros(shape = self.nVehiculos,dtype=np.int64)

        # Vaciamos los vehículos. 
        self.v_loads = np.zeros(shape=self.nVehiculos,dtype=np.int64) + self.v_maxCapacity
        self.v_loads = np.pad(self.v_loads, (0, self.nVehiculos - self.nVehiculos), 'constant', constant_values = 0)

        # Restauramos las demandas originales de los nodos.
        self.n_demands = copy.deepcopy(self.n_originalDemands)

        self.travelTime = np.zeros(shape=(self.nVehiculos), dtype = float)
        self.travelTime = np.pad(self.travelTime, (0, self.nVehiculos - self.nVehiculos), 'constant', constant_values = 999)

        # Calculamos las distancias que hay desde cada vehículo a cada nodo.
        self.n_distances = np.zeros(shape = (self.nVehiculos, self.nNodos))

        for i in range(0, self.nVehiculos):
            self.n_distances[i] = self.distanceMatrix[0]

        # Creamos un conjunto de rutas nuevo
        self.rutas = Routes(self.nVehiculos, self.nNodos, self.nVehiculos, self.nNodos, self.n_demands, self.n_coordenadas, self.v_speeds)

        # Creamos una nueva lista que almacena el orden en el que se visitan los nodos.
        self.v_ordenVisitas = []
        # Añadimos nuevas listas que tienen ya un 0 (depot) al comienzo, tantas como vehículos haya.
        for _ in range(self.nVehiculos):
            self.v_ordenVisitas.append([0])

        self.currEpisodeSteps = 0
               
        return self._get_obs(), self._get_info() # Con esto si quitamos el parámetro apply_api_compatibility 

    """
    Método que comprueba la validez de una acción. La acción será incorrecta si:
    - Se visita un nodo ya visitado
    - El vehículo no se mueve.
    - Se visita un nodo con una demanda superior a la que puede llevar el vehículo.
    - El vehículo llega al nodo antes del inicio de la ventana de tiempo o después del cierre de esta.
    """
    def checkAction(self, action, vehiculo):
        if self.visited[action] == 1:
            return False
        
        if self.v_posicionActual[vehiculo] == action:
            return False
        
        if self.v_loads[vehiculo] - self.n_demands[action] < 0:
            return False

        if self.n_twMin[action] > self.travelTime[vehiculo]:
            #print("MIN: {} - {}".format(self.minTW[action], self.currTime[vehiculo]))
            return False

        if self.n_twMax[action] < self.travelTime[vehiculo]:
            #print("MAX: {} - {}".format(self.maxTW[action], self.currTime[vehiculo]))
            return False

        return True
    

    # Método que obtiene las observaciones del entorno
    def _get_obs(self):
        obs = dict()
        obs["n_visited"] = self.visited
        obs["v_curr_position"] = self.v_posicionActual
        obs["v_loads"] = self.v_loads
        obs["n_demands"] = self.n_demands 
        obs["n_distances"] = self.n_distances.flatten() # Como es una matriz multidimensional hay que aplanarla

        return obs

    def _get_info(self):
        info = {'info' : 'None'}
        return info

    # Truncated se usa para indicar si el episodio ha finalizado porque se ha cumplido una condición fuera del
    # objetivo final del episodio, como haber alcanzado cierto límite de tiempo.
    def isTruncated(self):
        if self.maxSteps is None:
            return False

        return True if self.maxSteps <= self.currEpisodeSteps else False


    # Método que comprueba que haya finalizado el episodio
    def isDone(self): 
        if self.multiTrip:
            allVisited = np.all(self.visited[1:] == 1) # Con multitrip, solo miramos que se hayan visitado todos los nodos

            # Si se han visitado, damos el episodio por acabado
            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas) # Guardamos siempre el último conjunto de rutas completas, para poder dibujarlas al finalizar el entrenamiento.
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas) # Lo mismo con el orden de visitas
                self.tiempoFinal = copy.deepcopy(self.travelTime) # Y con los tiempos
                return True
        
        # De normal, todos los vehículos deben estar en el depot y todos los nodos deben haber sido visitados.
        if np.all(self.v_posicionActual == 0):
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas) # Guardamos siempre el último conjunto de rutas completas, para poder dibujarlas al finalizar el entrenamiento.
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas) # Lo mismo con el orden de visitas
                self.tiempoFinal = copy.deepcopy(self.travelTime) # Y con los tiempos
                return True
        
        # Si se han visitado todos los nodos pero no han regresado los vehículos al depot, este se pone como no visitado
        else:
            allVisited = np.all(self.visited == 1)
            if allVisited:
                self.visited[0] = 0

        return False

    """
    Método que añade los parámetros iniciales del método increasingIsDone.
    totalSteps: número total de pasos que durará el entrenamiento.
    minNumVisited: porcentaje mínimo de nodos que debe ser visitado al inicio del entrenamiento.
    increaseStart: porcentaje de pasos que deben haber pasado para que el porcentaje mínimo de nodos comience a aumentar.
    increaseRate: cuánto aumentará el porcentaje mínimo de nodos a visitar.
    everyNtimesteps: cada cuánto aumentará el porcentaje mínimo de nodos a visitar.
    """
    def setIncreasingIsDone(self, totalSteps, minNumVisited = 0.5, increaseStart = 0.5, increaseRate = 0.1, everyNtimesteps = 0.1):
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

        # Si se permiten múltiples viajes, se tendrá que revisar que ha finalizado el episodio aunque los vehículos no estén en el depot
        if self.multiTrip:
            porcentajeVisitados = np.count_nonzero(self.visited[1:self.nNodos] == 1) / self.nNodos

            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.travelTime)
                return True
    
        # Si no es multitrip, se tendrá que comprobar que todos han regresado al depot y que cumplen el porcentaje mínimo.
        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.travelTime)
                return True
        
        # Si no están todos en el depot, se marca el depósito como no visitado para que el/los vehículos que queden puedan regresar.
        else:
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.visited[0] = 0

        return False


    """
    Método que añade los parámetros iniciales del método decayingIsDone.
    totalSteps: número total de pasos que durará el entrenamiento.
    decayingStart: porcentaje de pasos que deben haber pasado para que el porcentaje mínimo de nodos comience a disminuir.
    decayingRate: cuánto disminuirá el porcentaje mínimo de nodos a visitar.
    everyNtimesteps: cada cuánto disminuirá el porcentaje mínimo de nodos a visitar.
    """
    def setDecayingIsDone(self, totalSteps, decayingStart = 0.5, decayingRate = 0.1, everyNtimesteps = 0.05):
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

        # Si tenemos multiTrip, entonces no es necesario que todos los vehículos hayan regresado al depot.
        if self.multiTrip:
            porcentajeVisitados = np.count_nonzero(self.visited[1:self.nNodos] == 1) / self.nNodos

            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.travelTime)
                return True
        
        # Si no es multitrip, solo comprobaremos si ha finalizado el episodio si los vehículos están en el depot
        if np.all(self.v_posicionActual == 0):
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
                        
            if porcentajeVisitados >= self.minimumVisited:
                self.grafoCompletado = copy.deepcopy(self.rutas)
                self.ordenVisitasCompletas = copy.deepcopy(self.v_ordenVisitas)
                self.tiempoFinal = copy.deepcopy(self.travelTime)
                return True
        
        # Si no lo están, marcamos el depot como no visitado 
        else:
            porcentajeVisitados = np.count_nonzero(self.visited[:self.nNodos] == 1) / self.nNodos
            
            if porcentajeVisitados >= self.minimumVisited:
                self.visited[0] = 0

     
        return False
    
    def createMatrix(self, value, shape):
        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, list):
            return np.array(value)
        
        return np.zeros(shape) + value

        
    # Método que calcula la recompensa a dar al agente.
    def getReward(self, distancia, action, vehicle):
        # Si no se mueve ningún vehículo, penalización
        if distancia == 0:
            return -1

        # Cuando el vehículo termina su ruta recibe una recompensa inversamente proporcional a lo lleno que vaya el vehículo,
        # a más llenado más recompensa. Por defecto v_loads está a 100 (capacidad máxima), y se le va restando según se recogen pedidos.
        if action % self.nNodos == 0:
            if self.v_loads[vehicle] != 0:
                return round(1/abs(self.v_loads[vehicle]), 2) 
            return 1

        # La recompensa será inversamente proporcional a la distancia recorrida, a mayor distancia, menor recompensa
        return round(1/abs(distancia), 2)


    # Calculamos cuánto tiempo queda hasta que cierren las ventanas de tiempo.
    # Current time tiene shape (nVehiculos,), mientras que el repeat devuelve shape (nNodos, nvehiculos)
    # No se puede hacer broadcast de esos rangos, pero con np.array([self.currTime]).T tenemos que curr time tiene shape (nVehiculos,1)
    # y con eso sí se puede hacer la resta que queremos
    def getTimeLeftTWClose(self):
        twClose = np.repeat([self.n_twMax], repeats=self.nVehiculos, axis=0) - np.array([self.travelTime]).T

        return twClose


    # Creamos las matrices de distancias y de tiempo. 
    # Estas matrices tendrán ya calculados la distancia y el tiempo que hay entre cada par de nodos, de manera que en el resto del código,
    # para saber estos datos, bastará con consultar las matrices, ahorrando cálculos.
    def createMatrixes(self):
        self.distanceMatrix = haversine_distances(X = self.n_coordenadas*np.pi/180,
                                                  Y = self.n_coordenadas*np.pi/180)

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
        self.v_loads = self.dataGen.vehicleInfo["maxCapacity"].to_numpy()
        self.v_maxCapacity = self.v_loads.max()
        self.v_speeds = self.dataGen.vehicleInfo["speed"].to_numpy()

        # Ventanas de tiempo
        self.n_twMin = self.dataGen.nodeInfo["minTW"].to_numpy()
        self.n_twMax = self.dataGen.nodeInfo["maxTW"].to_numpy()

    def renderRoutes(self, dir = 'default'):      
        if self.grafoCompletado == None:
            return
        
        # Llama a un método de guardado o a otro dependiendo de si se quieren todas las rutas en un mismo plot o no
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot(dir)

        else:
            self.grafoCompletado.guardarGrafos(dir)

        self.crearReport(self.ordenVisitasCompletas, self.tiempoFinal, directorio = dir)


    # Crea y guarda una imagen y un report el último conjunto de grafos completado 
    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.num_columns = min(self.nVehiculos, 4)
            self.num_rows = np.ceil(self.nVehiculos / self.num_columns).astype(int)

        plt.clf()

        for n, idGrafo in enumerate(range(len(self.rutas.grafos))):
            ax = plt.subplot(self.num_rows, self.num_columns, n + 1)

            self.rutas.grafos[idGrafo].dibujarGrafo(ax = ax)

        plt.pause(0.2) # PROBAR con esto: https://graphviz.readthedocs.io/en/stable/manual.html


    def close(self):
        super().close()
        
        if self.graphSavePath:
            self.generateReport(self.graphSavePath)
        else:
            self.generateReport('reporting')


    # Guarda el conjunto actual de grafos, independientemente de si están completos o no
    def generateReport(self, dir = 'default'):
        if self.grafoCompletado == None:
            return
        
        # Llama a un método de guardado o a otro dependiendo de si se quieren todas las rutas en un mismo plot o no
        if self.singlePlot:
            self.grafoCompletado.guardarGrafosSinglePlot(directorio = dir)

        else:
            self.grafoCompletado.guardarGrafos(directorio = dir)

        self.crearReport(self.ordenVisitasCompletas, self.tiempoFinal, directorio = dir)


    # Método que crea un pequeño report sobre las rutas obtenidas.
    # Simplemente crea un fichero con las rutas creadas mostrando el orden de las visitas, la duración de cada ruta y la duración
    # total de todas las rutas.
    def crearReport(self, v_ordenVisitas, travelTime, fecha = None, directorio = 'reports', name = 'report', extension = '.txt'):
        if fecha is None:
            fecha = str(date.today())

        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreDoc = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        tiempoTotal = 0

        with open(nombreDoc, 'w', encoding="utf-8") as f:
            f.write("############")
            f.write(str(date.today()))
            f.write("############")
            f.write("\n\nNúmero de vehíclos utilizados: {}".format(self.nVehiculos))
            f.write("\n")

            for ruta, tiempo in zip(v_ordenVisitas, travelTime):
                tiempoTotal += tiempo
                f.write("\n"+str(ruta) + " - t: " + str(round(tiempo, 2)) + " min")

            f.write("\n\nTiempo total: " + str(round(tiempoTotal, 2)) + " min")

            f.close()

    # Hace los cálculos necesarios para pasar de una acción a un vehículo y un nodo a visitar, y lo devuelve como string. No se usa.
    def actionParser(self, action):
        vehiculo = action // self.nNodos        
        action = action % self.nNodos

        return str(vehiculo) + "_" + str(action)
    
    # Para pasar el tiempo a horas y minutos devolverlo como string. No se usa. Creo.
    def minToStr(self, time):
        hora = time / 60
        minutos = time // 60

        return str(hora) + ":" + str(minutos)
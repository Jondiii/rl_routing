import pandas as pd
import numpy as np
import errno
import json
import sys
import os
import io


class DataReader:
    nameIndex = 'index'
    nameXcoord = 'coordenadas_X'
    nameYcoord = 'coordenadas_Y'
    nameDemands = 'demandas'
    nameMaxDemands = 'maxDemand'
    nameTWStart = 'minTW'
    nameTWEnd = 'maxTW'


    def __init__(self,
                 dataPath,
                 nodeFile="nodes",
                 vehicleFile="vehicles",
                 fileFormat = ".csv"
                 ):
        
        # Comprobamos si el directorio especificado en 'dataPath' existe
        if not os.path.exists(dataPath):
            print("Unable to find path: {}".format(dataPath), file=sys.stderr)
            sys.exit(1)  # Si no existe, salimos del programa

        self.nodePath = os.path.join(dataPath, nodeFile + fileFormat)
        self.vehiclePath = os.path.join(dataPath, vehicleFile + fileFormat)

        # Leer la información de nodos y vehículos
        if fileFormat == ".csv":
            self.nodeInfo = pd.read_csv(self.nodePath, index_col=0)
            self.vehicleInfo = pd.read_csv(self.vehiclePath, index_col=0)

        # Se usa para los benchmarks        
        elif fileFormat == ".json":
            with io.open(self.nodePath, 'rt', newline='') as file:
                jsonData = json.load(file)
            
            file.close()

            self.loadInfoJson(jsonData)

        else:
            print("The specified file format {} is not supported.".format(fileFormat), file=sys.stderr)


    def loadInfoJson(self, data):
        self.nodeInfo = self.loadNodeInfoJson(data, 'customer')
        self.vehicleInfo = self.loadVehicleInfoJson(data)



    def loadNodeInfoJson(self, data, keyword):
        nodeData = dict()
        self.nodeInfo = []
        
        value = data['depart']
        value['coordenadas_X'] = value['coordinates']['x']
        value['coordenadas_Y'] = value['coordinates']['y']

        self.nodeInfo.append(data['depart'])
        
        for key, value in data.items():
            if keyword in key:
                value['coordenadas_X'] = value['coordinates']['x']
                value['coordenadas_Y'] = value['coordinates']['y']
                self.nodeInfo.append(value)

        self.nodeInfo = pd.DataFrame.from_dict(self.nodeInfo)

        nodeData['index'] = self.nodeInfo.index

        nodeData['coordenadas_X'] = self.nodeInfo['coordenadas_X']
        nodeData['coordenadas_Y'] = self.nodeInfo['coordenadas_Y']
        
        nodeData['demandas'] = self.nodeInfo['demand']
        nodeData['maxDemand'] = self.nodeInfo['demand'].max()
        
        nodeData['minTW'] = self.nodeInfo["ready_time"].to_numpy()
        nodeData['maxTW'] = self.nodeInfo["due_time"].to_numpy()

        nodeData['service_time'] = self.nodeInfo["service_time"].to_numpy()

        return nodeData
    

    def loadVehicleInfoJson(self, data):
        vehicleData = pd.DataFrame(np.zeros(data['max_vehicle_number']))

        vehicleData.drop(columns=0, inplace=True)

        vehicleData['index'] = vehicleData.index

        vehicleData['initialPosition'] = 0
        vehicleData['maxCapacity'] = int(data["vehicle_capacity"])
        vehicleData['speed'] = 70

    
        return vehicleData.to_dict()



    def loadNodeData(self):
        nodeData = dict()

        nodeData['coordenadas_X'] = self.nodeInfo["coordenadas_X"]
        nodeData['coordenadas_Y'] = self.nodeInfo["coordenadas_Y"]
        nodeData['demandas'] = self.nodeInfo["demandas"]
        nodeData['maxDemand']= self.nodeInfo["maxDemand"]

        nodeData['minTW'] = self.nodeInfo["minTW"]
        nodeData['maxTW'] = self.nodeInfo["maxTW"]

        return nodeData


    def loadVehicleData(self):
        vehicleData = dict()

        vehicleData['maxCapacity'] = self.vehicleInfo["maxCapacity"]
        vehicleData['speed'] = self.vehicleInfo["speed"]
        vehicleData['maxCapacity'] = self.vehicleInfo["maxCapacity"]
        vehicleData['speed'] = self.vehicleInfo["speed"]

        return vehicleData
    

    def save_to_csv(self, path, nodeFile, vehicleFile):
        if not os.path.exists(path):
            os.makedirs(path)
        
        pd.DataFrame(self.nodeInfo).set_index('index').to_csv(os.path.join(path, nodeFile + '.csv'))
        pd.DataFrame(self.vehicleInfo).set_index('index').to_csv(os.path.join(path, vehicleFile + '.csv'))

    
    def make_node_csv_readable(self, nodeFile, coord_x, coord_y, demand, minTw, maxTw, maxDemand = None):
        if not os.path.exists(nodeFile):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), nodeFile)
        
        nodeData = pd.read_csv(nodeFile)
        nodeData.drop('CUST NO.', axis = 1, inplace=True)

        nodeData.index.names = [self.nameIndex ]

        nodeData.rename(columns = {coord_x : self.nameXcoord,
                                   coord_y : self.nameYcoord,
                                   demand : self.nameDemands,
                                   minTw : self.nameTWStart,
                                   maxTw : self.nameTWEnd},
                        inplace = True)

        if maxDemand is None:
            nodeData[self.nameMaxDemands] = nodeData[self.nameDemands].max()
        
        else:
            nodeData.rename(columns = {maxDemand : self.nameMaxDemands})

        nodeData.to_csv(nodeFile, columns=[self.nameXcoord,self.nameYcoord,self.nameDemands,self.nameTWStart,self.nameTWEnd,self.nameMaxDemands])


if __name__ == "__main__":
    reader = DataReader('data')

    base_dir = "data/solomon_dataset"
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv") and "vehicles" not in file:
                file_path = os.path.join(root, file)

                # Leer el CSV
                df = pd.read_csv(file_path)

                # Determinar el valor para la columna service_time
                if file.startswith('R'):
                    service_time_value = 10
                else:
                    service_time_value = 90

                # Añadir la columna service_time con el valor adecuado
                df['service_time'] = service_time_value

                # Asignar el valor 0 en la primera fila de la columna service_time
                df.at[0, 'service_time'] = 0

                # Guardar el CSV con las modificaciones
                df.to_csv(file_path, index=False)

                print(f"Modificado {file_path}")

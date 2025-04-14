import numpy as np
from .route import Route
from datetime import date
import matplotlib.pyplot as plt
import time
import os


class Solution:
    """
    Esta clase almacena la solución generada para un problema.
    Las rutas se almacenan de manera individual dentro de esta misma clase en formato de grafos, una clase
    también implementada en este proyecto.
    """

    def __init__(self, nNodos, coordenadas):
        #matplotlib.use('Agg') # Descomentar si se está trabajando en el server. Como esta clase se usa para visualizar rutas, si no se pone esto al hacer pruebas en el servidor, este peta.
        
        self.nNodos = nNodos  
        self.coordenadas = coordenadas[:nNodos]

        self.rutas = [Route(self.nNodos, self.coordenadas)] # Lista que contendrá las rutas individuales del problema, en forma de grafos.


    # Por cada vehículo, creamos un grafo.
    def nuevaRuta(self):
        self.rutas.append(Route(self.nNodos, self.coordenadas))


    # Hace que un vehículo vaya desde nodo 1 a nodo 2 y marque a nodo 2 como visitado.
    def visitEdge(self, nodo1, nodo2, distance, serviceTime):
        self.rutas[-1].visitEdge(nodo1, nodo2, distance, serviceTime)
        
    
    # Guarda una representación visual de los grafos (rutas) obtenidos. # TODO sacar esto de aquí y meterlo en la clase principal. O en training maanger
    def guardarGrafoSolucion(self, directorio, name = 'fig', extension = '.png'):
        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Se define cómo organizar los grafos. Máximo habrá 3 columnas, y tantas filas como sea necesario.
        num_columns = min(len(self.rutas), 3)
        num_rows = np.ceil(len(self.rutas) / num_columns).astype(int)

        # Se crea la figura y se asignan sus dimensiones.
        plt.clf()
        plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        # En cada sección de la figura se crea un subplot, que contendrá un único grafo.
        for n, idGrafo in enumerate(range(len(self.rutas))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.rutas[idGrafo].dibujarGrafo(ax = ax)

        # Este trozo de código añade números incrementales en función del número de imágenes que ya se hayan guardado en el propio directorio.
        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        # Guardamos el plot y lo cerramos
        plt.savefig(nombreFigura)
        plt.close()
    

    # Método que hace lo mismo que el anterior, solo que en vez de dibujar cada ruta (grafo) por separado,
    # se dibujan todas a la vez, en un único plot. Suele quedar un poco caótico.
    def guardarGrafosSinglePlot(self, fecha = None, directorio = 'grafos', name = 'fig', extension = '.png'):
        if fecha is None:
            fecha = str(date.today())
    
        directorio = os.path.join(directorio, fecha)

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Realmente tener tantas rutas pintadas a la vez es bastante confuso, por lo que no me he molestado en poner más de 10 colores.
        colorList = [
            "tab:red",
            "tab:orange",
            "tab:olive",
            "tab:green",
            "tab:cyan",
            "tab:blue",
            "tab:purple",
            "tab:pink",
            "tab:brown",
            "tab:gray"
        ]  # Para más info sobre colores: https://matplotlib.org/stable/tutorials/colors/colors.html

        plt.clf()
        ax = plt.subplot(1, 1, 1)
        
        # Pintamos cada ruta con un color distinto, sobre el mismo plot.
        for idGrafo, color in zip(range(len(self.rutas)), colorList):
            self.rutas[idGrafo].dibujarGrafo(ax = ax, edgeColor = color)

        # Este trozo de código añade números incrementales en función del número de imágenes que ya se hayan guardado en el propio directorio.
        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreFigura = os.path.join(directorio, name + '_' + siguiente_numero + extension)

        plt.savefig(nombreFigura)

        plt.close()


    def guardarTextoSolucion(self, directorio, name = 'report', extension = '.txt'):
        if not os.path.exists(directorio):
            os.makedirs(directorio)

        existentes = os.listdir(directorio)
        numeros = [int(nombre.split('_')[-1].split('.')[0]) for nombre in existentes
               if nombre.startswith(name + '_') and nombre.endswith(extension)]
        siguiente_numero = str(max(numeros) + 1 if numeros else 1)

        nombreDoc = os.path.join(directorio, name + '_' + siguiente_numero + extension)


        with open(nombreDoc, 'w', encoding="utf-8") as f:
            f.write("############")
            f.write(str(date.today()))
            f.write("############")
            f.write("\n\nNúmero de vehíclos utilizados: {}".format(len(self.rutas)))
            f.write("\n")

            travelDistanceTotal = self.getTotalDistance()

            for ruta in self.rutas:
                f.write("\n"+str(ruta.visitOrder) + " - distance: " + str(round(ruta.travelDistance, 2)))


            f.write("\n\nDistancia total: " + str(travelDistanceTotal))

            f.close()

    def getTotalDistance(self):
        travelDistanceTotal = 0.0

        for ruta in self.rutas:
            travelDistanceTotal += ruta.travelDistance

        return round(travelDistanceTotal, 2)

    # TODO / WIP. La idea de esto es que una ventana vaya mostrando las rutas según estas se van creando. Es decir, que primero se muestre
    # el depot, luego en otro frame se pinte el primer nodo a visitar, luego el segundo, etc.
    def getRutasVisual(self):
        num_columns = min(len(self.rutas), 3)
        num_rows = np.ceil(len(self.rutas) / num_columns).astype(int)

        plt.ion()

        fig = plt.figure()

        for n, idGrafo in enumerate(range(len(self.rutas))):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.rutas[idGrafo].dibujarGrafo(ax = ax)
        
        plt.pause(1)
        
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(1)



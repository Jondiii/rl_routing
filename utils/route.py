import networkx as nx
import numpy as np

# Esta clase es la encargada de representar rutas de manera individual.
class Route:
    """
    Recibe el número de nodos, sus demandas, las coordenadas y la velocidad del vehículo.
    """
    def __init__(self, nNodos, coordenadas) -> None:
        self.nNodos = nNodos

        # Creamos un grafo completo (es decir, un grafo donde todos los nodos están conectados entre sí)
        self.graph = nx.complete_graph(nNodos)
        self.coordenadas = {   # Enumerate pone números delante de los elementos de una lista, así que la i
                               # se queda con el número que haya puesto el enumerate.
            i: coordinates for i, coordinates in enumerate(coordenadas)
        }
        
        # Añadimos las coordenadas de cada nodo
        nx.set_node_attributes(self.graph, self.coordenadas, "coordinates")

        # Tenemos que especificar qué nodos tienen la propiedad de depot. Esto se hace creando una lista donde todos
        # los nodos tengan un 0 en depot, salvo el primer nodo, que sí será el depot y tendrá un 1.
        depotList = np.zeros(nNodos)
        depotList[0] = 1 # Marcamos como depot el primer nodo de la lista 
        depotDict = {i: depot for i, depot in enumerate(depotList)}
        nx.set_node_attributes(self.graph, depotDict, "depot")
        
        # Marcamos todos los nodos como no visitados
        nx.set_node_attributes(self.graph, False, "visited")

        # Pintamos los nodos de negro, salvo el depot, que será rojo.
        nx.set_node_attributes(self.graph, "black", "node_color")
        self.graph.nodes[0]["node_color"] = "red"

        # Definimos atributos del grafo. Marcamos los arcos como no visitados.
        nx.set_edge_attributes(self.graph, False, "visited")

        # Offset a usar a la hora de dibujar labels. Esto hace que se seleccione un valor aleatorio entre 0 y 0.045
        # con el que se desplazará ligeramente el lugar en el que se dibujan los labels, para que no queden justo
        # en medio del nodo.
        self.offset = np.array([0, 0.045])

    # Método que hace que un nodo sea visitado, devolviendo información sobre la distancia recorrida y el tiempo empleado.
    def visitEdge(self, sourceNode, targetNode):
        if sourceNode == targetNode:
            return
        
        self.graph.edges[sourceNode, targetNode]["visited"] = True
        self.graph.nodes[targetNode]["visited"] = True

    # Método encargado de dibujar una ruta concreta.
    def dibujarGrafo(self, ax, edgeColor = "red"):
        # Método que comprueba si un nodo ha sido visitado o no.
        def isNodeVisited(node):
            return (self.graph.nodes[node]["visited"] == True) or (self.graph.nodes[node]["depot"] == True)
        
        posicion = nx.get_node_attributes(self.graph, "coordinates")

        # Primero dibujamos los nodos en base al método creado. Solo se usarán los métodos que devuelvan True.
        subGrafo = nx.subgraph_view(self.graph, filter_node = isNodeVisited)

        # Seleccionamos los colores de los nodos (negro)
        node_colors = [c for _, c in subGrafo.nodes(data="node_color")]

        # Pinta todos los nodos a la vez.
        nx.draw_networkx_nodes(
            self.graph,
            posicion,
            node_color = node_colors,
            ax=ax,
            nodelist = subGrafo.nodes(),
            node_size=100
            )

        # Después dibujamos las aristas/arcos que hayan sido visitados
        aristas = [x for x in self.graph.edges(data = True) if x[2]["visited"]]

        nx.draw_networkx_edges(
            self.graph,
            posicion,
            alpha=0.5,
            edgelist=aristas,
            edge_color=edgeColor,
            ax=ax,
            width=1.5,
        )

        # Ponemos el número de cada nodo
        posicionIDNodo = {k: (v) for k, v in posicion.items()}
        labelIDNodo = {id: id for id, _ in subGrafo.nodes.items()}
        nx.draw_networkx_labels(
            self.graph, posicionIDNodo, labels=labelIDNodo, ax=ax, font_color = "white", font_size = 8
        )

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
from typing import Union, Tuple, List, Iterable, Any
import os

class DataTriangle:
    def __init__(self, eps=10e-3):
        self.triang = None
        self.interpolator = None
        self.x = None
        self.y = None
        self.values = None
        self.sqrt3 = np.sqrt(3)
        self.eps = eps

    def ternary_to_cartesian(self, coordinates):
        a = coordinates[:, 0]
        b = coordinates[:, 1]
        c = coordinates[:, 2]
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = self.sqrt3 * 0.5 * c / (a + b + c)
        return x, y

    def format_data(self, coordinates, values):
        coordinates = np.array(coordinates).reshape(-1, 3)
        coordinates = coordinates/coordinates.sum(axis=-1, keepdims=True)
        values = np.array(values)
        return coordinates, values

    def format_coordinates(self, coordinates):
        coordinates = np.array(coordinates).reshape(-1, 3)
        coordinates = coordinates/coordinates.sum(axis=-1, keepdims=True)
        return coordinates

    def triangulate_data(self, coordinates, values):
        coordinates, values = self.format_data(coordinates, values)
        self.x, self.y = self.ternary_to_cartesian(coordinates)
        self.values = values

        # Setup for nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array([self.x, self.y]).T)

        # Artificially add points along the edges of the domain
        self.add_side_examples(nbrs)

        # Triangulation
        self.triang = Triangulation(self.x, self.y)
        self.interpolator = LinearTriInterpolator(self.triang, self.values)

    def triangulate_data_dict(self, data_dict):
        coordinates = []
        values = []
        for k, v in data_dict.items():
            coordinates.append(k)
            values.append(v)
        self.triangulate_data(coordinates, values)

    def add_side_examples(self, nbrs):
        x_edge_left = np.linspace(0-self.eps, 0.5, 55)
        x_edge_right = np.linspace(0.5, 1+self.eps, 55)
        x_edge_bottom = np.linspace(0-self.eps, 1+self.eps, 55)

        y_edge_bottom = np.zeros_like(x_edge_bottom)
        y_edge_left = self.sqrt3 * x_edge_left
        y_edge_right = self.sqrt3 * (1 - x_edge_right)

        values_edge_bottom = self.values[
            nbrs.kneighbors(np.array([x_edge_bottom, y_edge_bottom]).T, return_distance=False).ravel()]
        values_edge_left = self.values[
            nbrs.kneighbors(np.array([x_edge_left, y_edge_left]).T, return_distance=False).ravel()]
        values_edge_right = self.values[
            nbrs.kneighbors(np.array([x_edge_right, y_edge_right]).T, return_distance=False).ravel()]

        self.x = np.concatenate((self.x, x_edge_bottom, x_edge_left, x_edge_right))
        self.y = np.concatenate((self.y, y_edge_bottom, y_edge_left, y_edge_right))
        self.values = np.concatenate((self.values, values_edge_bottom, values_edge_left, values_edge_right))

    def plot(self, savedir=None, label='', show=False):
        # Triangle border
        triangle = Polygon([[0, 0], [0.5, self.sqrt3 / 2], [1, 0]], closed=True, fill=False, edgecolor='k')

        plt.figure()
        plt.gca().add_patch(triangle)
        plt.plot(self.x, self.y, 'ko', markersize=2)
        Xi, Yi = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, self.sqrt3 / 2, 200))
        Zi = self.interpolator(Xi, Yi)
        im = plt.imshow(Zi, extent=(0, 1, 0, self.sqrt3 / 2), origin='lower', cmap=cm.viridis)
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, self.sqrt3 / 2)
        cbar = plt.colorbar(im)
        cbar.set_label('Interpolated Value', rotation=270, labelpad=15)
        if show:
            plt.show()
        if savedir:
            plt.savefig(os.path.join(savedir, 'dt_plot' + label))

    def plot_against(self, other, savedir=None, label='', show=False):
        # Triangle border
        triangle = Polygon([[0, 0], [0.5, self.sqrt3 / 2], [1, 0]], closed=True, fill=False, edgecolor='k')

        plt.figure()
        plt.gca().add_patch(triangle)
        plt.plot(self.x, self.y, 'ko', markersize=2)
        Xi, Yi = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, self.sqrt3 / 2, 200))
        Zi = self.interpolator(Xi, Yi)
        oZi = other.interpolator(Xi, Yi)
        im = plt.imshow(np.maximum(Zi-oZi, 0), extent=(0, 1, 0, self.sqrt3 / 2), origin='lower', cmap=cm.viridis)
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, self.sqrt3 / 2)
        cbar = plt.colorbar(im)
        cbar.set_label('Interpolated Value', rotation=270, labelpad=15)
        if show:
            plt.show()
        if savedir:
            plt.savefig(os.path.join(savedir, 'dt_plot' + label))

    def get_value(self, coordinates):
        coordinates = self.format_coordinates(coordinates)
        x, y = self.ternary_to_cartesian(coordinates)
        out = self.interpolator(x, y)
        return out



def ternary_to_cartesian(coordinates: np.ndarray) -> [np.ndarray, np.ndarray]:
    a = coordinates[:, 0]
    b = coordinates[:, 1]
    c = coordinates[:, 2]
    x = 0.5 * (2 * b + c) / (a + b + c)
    y = np.sqrt(3) * 0.5 * c / (a + b + c)
    return x, y


def plot_transitions(before: Iterable[Iterable[int]],
                     after: Iterable[Iterable[int]],
                     labels: Union[None, Iterable[Iterable[str]]] = None) -> None:
    # Check if lists are of the same length
    assert len(before) == len(after), "Before and after lists must have the same length"

    valid_indices = [i for i, (b, a) in enumerate(zip(before, after)) if all(n >= 0 for n in b + a)]

    # Filter out tuples with negative numbers and normalize the tuples
    before = [np.array(before[i]) / np.sum(before[i]) for i in valid_indices]
    after = [np.array(after[i]) / np.sum(after[i]) for i in valid_indices]

    # Convert the triangle coordinates to Cartesian coordinates
    before_x, before_y = ternary_to_cartesian(np.array(before))
    after_x, after_y = ternary_to_cartesian(np.array(after))

    # Triangle border
    triangle = Polygon([[0, 0], [0.5, np.sqrt(3) / 2], [1, 0]], closed=True, fill=False, edgecolor='k')

    plt.figure()
    plt.gca().add_patch(triangle)

    # Add labels
    if labels is not None:
        plt.text(-0.05, -0.05, labels[0])
        plt.text(1.05, -0.05, labels[1])
        plt.text(0.5, np.sqrt(3) / 2 + 0.05, labels[2])

    # Plot the starting points
    plt.plot(before_x, before_y, 'ro', markersize=5)

    # Plot the transitions
    for bx, by, ax, ay in zip(before_x, before_y, after_x, after_y):
        plt.arrow(bx, by, ax - bx, ay - by, head_width=0.02, head_length=0.02, fc='blue', ec='blue')

    plt.axis('equal')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, np.sqrt(3) / 2 + 0.1)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    n_points = 100
    data = np.random.rand(n_points, 3)  # Normalize points to the simplex
    values = np.random.rand(n_points)  # Generate new random color values for the points

    dt = DataTriangle()
    dt.triangulate_data(data, values)
    #dt.plot(show=False)
    points = np.array([[0.3, 0.4, 0.3],
                       [0.2, 0.2, 0.2]])

    print(dt.get_value(points))

    before = np.array([
                [0.48110827803611755, 0.015896467491984367, 0.5029952526092529],
                [0.28308770060539246, 0.5825887322425842, 0.13432355225086212],
                 [0.00040784297743812203, 0.5992700457572937, 0.4003221094608307],
                 [0.003993670921772718, 0.32172998785972595, 0.6742762923240662],
                 [0.5644619464874268, 0.026117807254195213, 0.4094202518463135],
                 [0.2613866329193115, 0.002713542664423585, 0.735899806022644],
                 [0.37652266025543213, 0.2658577859401703, 0.35761958360671997],
                 [0.0023109056055545807, 0.603257417678833, 0.3944316506385803],
                 [0.4177072048187256, 0.01164880022406578, 0.5706440806388855],
                 [0.3366726040840149, 0.6625211238861084, 0.0008062669658102095],
                 [0.33743903040885925, 0.1251264214515686, 0.5374345183372498],
                 [0.20123137533664703, 0.5323725938796997, 0.26639607548713684],
                 [0.11355769634246826, 0.561661422252655, 0.3247809112071991],
                 [0.2776881754398346, 0.11074990779161453, 0.6115615367889404],
                 [0.32861968874931335, 0.27471810579299927, 0.396662175655365],
                 [0.10392884165048599, 0.17007102072238922, 0.7260000705718994],
                 [0.4812844395637512, 0.23678763210773468, 0.28192776441574097],
                 [4.2463700083317235e-05, 0.572079598903656, 0.4278779923915863],
                 [0.09076990187168121, 0.2716616988182068, 0.6375683546066284],
                 [0.2761760354042053, 0.6979105472564697, 0.025913426652550697],
                 [0.24316367506980896, 0.6628663539886475, 0.09396997094154358],
                 [0.34603065252304077, 0.6491749882698059, 0.0047944048419594765],
                 [0.4075455963611603, 0.5024484992027283, 0.09000592678785324],
                 [0.18214763700962067, 0.03583827614784241, 0.782014012336731],
                 [9.937480353983119e-05, 0.3438945412635803, 0.6560060381889343],
                 [0.600994348526001, 0.11015129834413528, 0.28885436058044434]])

    after = [(24, 0, 30),
             (13, 40, 1),
             (0, 29, 25),
             (0, 18, 36),
             (28, 0, 26),
             (9, 0, 45),
             (20, 6, 28),
             (0, 33, 21),
             (20, 0, 34),
             (19, 35, 0),
             (16, 4, 34),
             (9, 31, 14),
             (5, 37, 12),
             (12, 0, 42),
             (14, 10, 30),
             (1, 10, 43),
             (25, 9, 20),
             (0, 31, 23),
             (2, 12, 40),
             (10, 44, 0),
             (9, 44, 1),
             (19, 35, 0),
             (20, 31, 3),
             (6, 0, 48),
             (0, 11, 43),
             (31, 3, 20)]

    plot_transitions(before, after, ['Cr', 'Fe', 'Ni'])
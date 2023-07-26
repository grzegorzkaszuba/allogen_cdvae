import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors
from typing import Union, Tuple, List, Iterable, Any

class DataTriangle:
    def __init__(self):
        self.triang = None
        self.interpolator = None
        self.x = None
        self.y = None
        self.values = None
        self.sqrt3 = np.sqrt(3)

    def ternary_to_cartesian(self, coordinates):
        a = coordinates[:, 0]
        b = coordinates[:, 1]
        c = coordinates[:, 2]
        x = 0.5 * (2 * b + c) / (a + b + c)
        y = self.sqrt3 * 0.5 * c / (a + b + c)
        return x, y

    def triangulate_data(self, coordinates, values):
        self.x, self.y = self.ternary_to_cartesian(coordinates)
        self.values = values

        # Setup for nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array([self.x, self.y]).T)

        # Artificially add points along the edges of the domain
        self.add_side_examples(nbrs)

        # Triangulation
        self.triang = Triangulation(self.x, self.y)
        self.interpolator = LinearTriInterpolator(self.triang, self.values)

    def add_side_examples(self, nbrs):
        x_edge_left = np.linspace(0, 0.5, 100)
        x_edge_right = np.linspace(0.5, 1, 100)
        x_edge_bottom = np.linspace(0, 1, 100)

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

    def show_plot(self):
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
        plt.show()

    def get_value(self, coordinates):
        x, y = self.ternary_to_cartesian(coordinates)
        return self.interpolator(x, y)


np.random.seed(0)
n_points = 100
data = np.random.rand(n_points, 3)
data /= np.sum(data, axis=1)[:, None]  # Normalize points to the simplex
values = np.random.rand(n_points)  # Generate new random color values for the points

dt = DataTriangle()
dt.triangulate_data(data, values)
dt.show_plot()
point = np.array([[0.3, 0.4, 0.3]])
print(dt.get_value(point))


def ternary_to_cartesian(coordinates: np.ndarray) -> [np.ndarray, np.ndarray]:
    a = coordinates[:, 0]
    b = coordinates[:, 1]
    c = coordinates[:, 2]
    x = 0.5 * (2 * b + c) / (a + b + c)
    y = np.sqrt(3) * 0.5 * c / (a + b + c)
    return x, y


def plot_transitions(before: Iterable[Iterable[int, int, int]],
                     after: Iterable[Iterable[int, int, int]],
                     labels: Union[None, Iterable[Iterable[str, str, str]]] = None) -> None:
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



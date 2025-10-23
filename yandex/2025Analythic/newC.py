import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def graph(class1, class2, a, b, c, linePoint, center):
    fig, ax = plt.subplots(figsize=(10, 8))
    class1 = class1.T
    class2 = class2.T
    ax.scatter(class1[0], class1[1], c="red", alpha=0.7, s=50)
    ax.scatter(class2[0], class2[1], c="blue", alpha=0.7, s=50)

    x_min, x_max = ax.get_xlim()
    x_line = np.linspace(x_min, x_max, 100)

    # Calculate y values for the line (ax + by + c = 0 => y = (-a*x - c)/b)
    if b != 0:
        y_line = (-a * x_line - c) / b
        ax.plot(x_line, y_line, "k-", linewidth=2, label=f"Line: {a}x + {b}y + {c} = 0")
    else:
        # Vertical line case (b = 0)
        x_vertical = -c / a if a != 0 else 0
        ax.axvline(x=x_vertical, color="k", linewidth=2, label=f"Line: {a}x + {c} = 0")

    ax.scatter(linePoint[0], linePoint[1], c="orange", alpha=0.7, s=50)
    ax.scatter(center[0], center[1], c="green", alpha=0.7, s=50)
    plt.savefig("C.png")
    pass


def min_distance_naive(red_points, blue_points, linePoint):
    """
    Find minimum distance between red and blue points using pairwise computation

    Args:
        red_points: 2xN array of red points
        blue_points: 2xM array of blue points
        linePoint: Point for the line

    Returns:
        min_dist: minimum distance between any red-blue pair
        min_pair: indices of the closest pair (red_index, blue_index)
    """
    # Compute all pairwise distances
    distances = distance.cdist(red_points, blue_points)

    # Find minimum distance and corresponding indices
    min_dist = np.min(distances)
    red_idx, blue_idx = np.unravel_index(np.argmin(distances), distances.shape)
    point1 = class1[red_idx]
    point2 = class2[blue_idx]
    center = (point1 + point2) / 2
    print(class1[red_idx], class2[blue_idx])
    a, b, c = line_by_ponints(linePoint, center)
    print(linePoint, center)
    print(a, b, c)
    minLineDist = (a * point2[0] + b * point2[1] + c) / (a * a + b * b)

    graph(class1, class2, a, b, c, linePoint, center)
    return min_dist, (red_idx, blue_idx), abs(minLineDist)


def line_by_ponints(point1, point2):
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = point1[0] * (point1[1] - point2[1]) + point1[1] * (point2[0] - point1[0])
    return a, b, c


with open("inputC.txt", "r") as file:
    lines = file.readlines()
# input first class of points
num1Class = int(lines[0])
points = []
class1 = np.zeros((num1Class, 2), dtype=float)
for i in range(num1Class):
    class1[i] = np.array(list(map(float, lines[i + 1].split())))
    points.append([class1[i][0], class1[i][1], -1])
# input second class of points
num2Class = int(lines[num1Class + 1])
class2 = np.zeros((num2Class, 2), dtype=float)
for i in range(num2Class):
    class2[i] = np.array(list(map(float, lines[i + num1Class + 2].split())))
    points.append([class2[i][0], class2[i][1], 1])
linePoint = np.array(list(map(float, lines[num2Class + num1Class + 2].split())))
points = np.array(points)
print("Points in cartesian coordinates:")
print(points)
print(class1, class2)
print("Transposed:")
print(class2.T)
for i in range(num1Class + num2Class):
    points[i][0] -= linePoint[0]
    points[i][1] -= linePoint[1]
print(min_distance_naive(class1, class2, linePoint))

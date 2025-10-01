import numpy as np


def car2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def arrayCar2pol(points, size):
    for i in range(size):
        points[i][0], points[i][1] = car2pol(points[i][0], points[i][1])
    points = points[points[:, 1].argsort()]
    return points


def find_line_angle_range(points, num1Class, num2Class):
    size = num1Class + num2Class
    numOfPoints = 0
    label = int(points[0][2])
    if label == 0:
        numClass = num1Class
    else:
        numClass = num2Class
    start = -1
    end = -1
    for i in range(size):
        if int(points[i][2]) != label:
            end = i - 1
            break
        numOfPoints += 1
    for i in range(size - 1, 0, -1):
        if int(points[i][2]) != label:
            start = i + 1
            if i == size - 1:
                start = 0
            break
        numOfPoints += 1
    if numOfPoints != numClass:
        return -1
    if end > start:
        class1ArcLength = points[end][1] - points[start][1]
    else:
        class1ArcLength = 2 * np.pi - (points[start][1] - points[end][1])
    print(f"Class {label}, length {class1ArcLength}")
    end2 = (start - 1) % size
    start2 = (end + 1) % size
    print(
        f"1st arc:({points[start][1] * 180 / np.pi},{points[end][1] * 180 / np.pi}), 2st arc:({points[start2][1] * 180 / np.pi},{points[end2][1] * 180 / np.pi})"
    )

    if end2 > start2:
        class2ArcLength = points[end2][1] - points[start2][1]
    else:
        class2ArcLength = 2 * np.pi - (points[start2][1] - points[end2][1])
    print(f"Class {1 - label}, length {class2ArcLength}")
    if class1ArcLength > np.pi or class2ArcLength > np.pi:
        return -1
    return points[start][1], points[end][1], points[start2][1], points[end2][1]


def line_limits(s1, e1, s2, e2):
    print(s1, e1, s2, e2)
    e1 -= s1
    s2 -= s1
    e2 -= s1
    if s2 < 0:
        s2 += 2 * np.pi
    if e1 < 0:
        e1 += 2 * np.pi
    if e2 < 0:
        e2 += 2 * np.pi
    print(0, e1, s2, e2)
    start = max(e2 - np.pi, e1)
    end = s2
    return start + s1, end + s1


with open("inputC.txt", "r") as file:
    lines = file.readlines()
# input first class of points
num1Class = int(lines[0])
points = []
class1 = np.zeros((num1Class, 2), dtype=float)
for i in range(num1Class):
    class1[i] = np.array(list(map(float, lines[i + 1].split())))
    points.append([class1[i][0], class1[i][1], 0])
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
for i in range(num1Class + num2Class):
    points[i][0] -= linePoint[0]
    points[i][1] -= linePoint[1]
points = arrayCar2pol(points, num1Class + num2Class)
print("Points in polar coordinates:")
print(points)
print(find_line_angle_range(points, num1Class, num2Class))
s1 = 0
s2 = 0
e1 = 0
e2 = 0
if find_line_angle_range(points, num1Class, num2Class) != -1:
    s1, e1, s2, e2 = find_line_angle_range(points, num1Class, num2Class)
print("First class points:")
for i in range(num1Class):
    print(class1[i])

print("Second class points:")
for i in range(num2Class):
    print(class2[i])
print(line_limits(s1, e1, s2, e2))

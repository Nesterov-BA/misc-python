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
    print(label, class1ArcLength)
    end2 = (start - 1) % size
    start2 = (end + 1) % size
    print(start, end, start2, end2)

    if end2 > start2:
        class2ArcLength = points[end2][1] - points[start2][1]
    else:
        class2ArcLength = 2 * np.pi - (points[start2][1] - points[end2][1])
    print(1 - label, class2ArcLength)
    if class1ArcLength > np.pi or class2ArcLength > np.pi:
        return -1


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
for i in range(num1Class + num2Class):
    points[i][0] -= linePoint[0]
    points[i][1] -= linePoint[1]
points = arrayCar2pol(points, num1Class + num2Class)
print(points)
print(find_line_angle_range(points, num1Class, num2Class))
print("First class points:")
for i in range(num1Class):
    print(class1[i])

print("Second class points:")
for i in range(num2Class):
    print(class2[i])

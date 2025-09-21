import numpy as np


def find_coefficients(point1, point2):
    if abs(point1[0] - point2[0]) < 0.00000001:
        A = 1
        B = 0
        C = -point1[0]
    elif abs(point1[1] - point2[1]) < 0.00000001:
        A = 0
        B = 1
        C = -point1[1]
    else:
        A = 1 / (point2[0] - point1[0])
        B = 1 / (point1[1] - point2[1])
        C = point1[0] / (point1[0] - point2[0]) + point1[1] / (point2[1] - point1[1])
        # B = B / A
        # C = C / A
        # A = 1
    return A, B, C


def line_divides_ponts(point1, point2, num1Class, num2Class, A, B, C):
    start1 = num1Class
    start2 = num2Class
    sign1 = 0
    sign2 = 0
    for i in range(num1Class):
        if abs(A * point1[i][0] + B * point1[i][1] + C) > 0.0000001:
            sign1 = np.sign(A * point1[i][0] + B * point1[i][1] + C)
            start1 = i
            break
    for i in range(num2Class):
        if abs(A * point2[i][0] + B * point2[i][1] + C) > 0.0000001:
            sign2 = np.sign(A * point2[i][0] + B * point2[i][1] + C)
            start2 = i
            break

    if start1 == num1Class and start2 == num2Class:
        print("All points are on the same line")
        return 0

    for i in range(start1 + 1, num1Class):
        print(A * point1[i][0] + B * point1[i][1] + C)
        if abs(A * point1[i][0] + B * point1[i][1] + C) < 0.0000001:
            continue
        if np.sign(A * point1[i][0] + B * point1[i][1] + C) != sign1:
            return -1
    for i in range(start2 + 1, num2Class):
        print(A * point2[i][0] + B * point2[i][1] + C)
        if abs(A * point2[i][0] + B * point2[i][1] + C) < 0.0000001:
            continue
        if np.sign(A * point2[i][0] + B * point2[i][1] + C) != sign2:
            return -1

    return 1


def find_edge_poins(points, linePoint, numPoints):
    for point in points:
        A, B, C = find_coefficients(point, linePoint)
        for i in range(numPoints):
            if abs(A * points[i][0] + B * points[i][1] + C) > 0.0000001:
                sign1 = np.sign(A * points[i][0] + B * points[i][1] + C)
                start1 = i
                break


with open("inputC.txt", "r") as file:
    lines = file.readlines()
# input first class of points
num1Class = int(lines[0])
class1 = np.zeros((2, num1Class), dtype=float)
for i in range(num1Class):
    class1[i] = np.array(list(map(float, lines[i + 1].split())))
# input second class of points
num2Class = int(lines[num1Class + 1])
class2 = np.zeros((2, num1Class), dtype=float)
for i in range(num2Class):
    class2[i] = np.array(list(map(float, lines[i + num1Class + 2].split())))
linePoint = np.array(list(map(float, lines[num2Class + num1Class + 2].split())))
print("First class points:")
for i in range(num1Class):
    print(class1[i])

print("Second class points:")
for i in range(num2Class):
    print(class2[i])

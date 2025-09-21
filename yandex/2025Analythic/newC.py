# import sys
import numpy as np
import math


def read_all():
    with open("inputC.txt", "r") as file:
        lines = file.readlines()
    # input first class of points
    num1Class = int(lines[0])
    class1 = np.zeros((num1Class, 2), dtype=float)
    for i in range(num1Class):
        class1[i] = np.array(list(map(float, lines[i + 1].split())))
    # input second class of points
    num2Class = int(lines[num1Class + 1])
    class2 = np.zeros((num2Class, 2), dtype=float)
    for i in range(num2Class):
        class2[i] = np.array(list(map(float, lines[i + num1Class + 2].split())))
    linePoint = np.array(list(map(float, lines[num2Class + num1Class + 2].split())))
    A = class1
    B = class2
    px = linePoint[0]
    py = linePoint[1]
    return A, B, (px, py)


# нормализация угла в [0, 2π)
TAU = 2.0 * math.pi


def norm_ang(a):
    a %= TAU
    if a < 0:
        a += TAU
    return a


# Пересечение двух множеств, каждое — объединение 1–2 интервалов в [0,2π)
def intersect_sets(S, T):
    # S, T — списки интервалов [L,R] с 0<=L<=R<=2π
    res = []
    for l1, r1 in S:
        for l2, r2 in T:
            l = max(l1, l2)
            r = min(r1, r2)
            if l <= r:
                res.append((l, r))
    # Упростим до максимум двух непересекающихся интервалов
    if not res:
        return []
    res.sort()
    merged = []
    cl, cr = res[0]
    for l, r in res[1:]:
        if l <= cr:
            cr = max(cr, r)
        else:
            merged.append((cl, cr))
            cl, cr = l, r
    merged.append((cl, cr))
    return merged


def add_arc(center, halfwidth):
    # Возвращает список 1–2 интервалов в [0,2π) для дуги [c-h, c+h] по окружности
    L = center - halfwidth
    R = center + halfwidth
    Lm = norm_ang(L)
    Rm = norm_ang(R)
    if Lm <= Rm:
        return [(Lm, Rm)]
    else:
        # Обернулось
        return [(0.0, Rm), (Lm, TAU)]


def feasible(A_polar, B_polar, m):
    # Проверяем, существует ли угол θ с зазором >= m
    S = [(0.0, TAU)]  # текущее допустимое множество для θ
    eps = 1e-15
    for r, phi in A_polar:
        t = m / r
        if t > 1 + 1e-15:
            return False
        t = min(max(t, 0.0), 1.0)
        alpha = math.acos(t)  # нужно cos(θ-phi) >= t
        arcs = add_arc(phi, alpha)
        S = intersect_sets(S, arcs)
        if not S:
            return False
    for r, phi in B_polar:
        t = m / r
        if t > 1 + 1e-15:
            return False
        t = min(max(t, 0.0), 1.0)
        alpha = math.acos(t)
        # для класса B нужно -cos(θ-phi) >= t  <=>  θ в окрестности (phi+π) с halfwidth=alpha
        arcs = add_arc(norm_ang(phi + math.pi), alpha)
        S = intersect_sets(S, arcs)
        if not S:
            return False
    # Нужна строгость (m>0 ⇒ автоматически cos>0/-cos>0), достаточно непустого множества
    # (если m==0, это проверка на разделимость)
    return True


def solve():
    A, B, P = read_all()
    px, py = P

    # Векторы от P к точкам, расстояния и углы
    A_polar = []
    B_polar = []

    # Если точка совпадает с P — разделить прямой через P строго невозможно
    for x, y in A:
        dx, dy = x - px, y - py
        r = math.hypot(dx, dy)
        if r == 0.0:
            print(-1)
            return
        phi = math.atan2(dy, dx)
        A_polar.append((r, norm_ang(phi)))
    for x, y in B:
        dx, dy = x - px, y - py
        r = math.hypot(dx, dy)
        if r == 0.0:
            print(-1)
            return
        phi = math.atan2(dy, dx)
        B_polar.append((r, norm_ang(phi)))

    # Сначала проверим разделимость (m -> 0)
    if not feasible(A_polar, B_polar, 0.0):
        print(-1)
        return

    # Верхняя граница: не больше минимального расстояния до P
    hi = min(min(r for r, _ in A_polar), min(r for r, _ in B_polar))
    lo = 0.0

    # Двоичный поиск максимального m
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if feasible(A_polar, B_polar, mid):
            lo = mid
        else:
            hi = mid

    # Вывод с нужной точностью
    print("{:.10f}".format(lo))


solve()

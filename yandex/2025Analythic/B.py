import numpy as np


def entropy(zeros, total):
    if zeros == total or zeros == 0:
        return 0
    else:
        p = zeros / total
        return -p * np.log(p) - (1 - p) * np.log(1 - p)


def count_leaves(numLeavesSt, childLeft, childRight):
    leavesNum = np.ones(2 * numLeavesSt - 1, dtype=int)
    for i in range(numLeavesSt, 2 * numLeavesSt - 1):
        leavesNum[i] = (
            leavesNum[childLeft[i - numLeavesSt] - 1]
            + leavesNum[childRight[i - numLeavesSt] - 1]
        )
    print(leavesNum)
    return leavesNum


def count_children(numLeavesSt, childLeft, childRight):
    childrenNum = np.zeros(2 * numLeavesSt - 1, dtype=int)
    for i in range(numLeavesSt, 2 * numLeavesSt - 1):
        childrenNum[i] = (
            childrenNum[childLeft[i - numLeavesSt] - 1]
            + childrenNum[childRight[i - numLeavesSt] - 1]
            + 2
        )
        print(childrenNum)
    return childrenNum


def count_ones(numLeavesSt, marks, childLeft, childRight):
    leavesNum = np.ones(2 * numLeavesSt - 1, dtype=int)
    for i in range(numLeavesSt):
        leavesNum[i] = marks[i]
    for i in range(numLeavesSt, 2 * numLeavesSt - 1):
        leavesNum[i] = (
            leavesNum[childLeft[i - numLeavesSt] - 1]
            + leavesNum[childRight[i - numLeavesSt] - 1]
        )
    print(leavesNum)
    return leavesNum


def prune_count_leaves(childLeft, childRight, numLeavesSt):
    leaves = count_leaves(numLeavesSt, childLeft, childRight)
    for index in range(2 * numLeavesSt - 1):
        leaves[index] = leaves[2 * numLeavesSt - 2] - leaves[index] + 1
    return leaves


def entropies(numLeavesSt, marks, childLeft, childRight):
    onesNum = count_ones(numLeavesSt, marks, childLeft, childRight)
    leavesNum = count_leaves(numLeavesSt, childLeft, childRight)
    entropies = np.zeros(2 * numLeavesSt - 1, dtype=float)
    for i in range(2 * numLeavesSt - 1):
        entropies[i] = entropy(onesNum[i], leavesNum[i])
    print(entropies)
    return entropies


with open("inputB.txt", "r") as file:
    lines = file.readlines()

numLeavesSt, numLeavesEnd = list(map(int, lines[0].split()))
marks = np.array(list(map(int, lines[1].split())))
childLeft = np.zeros(numLeavesSt - 1, dtype=int)
childRight = np.zeros(numLeavesSt - 1, dtype=int)
for i in range(numLeavesSt - 1):
    childLeft[i], childRight[i] = list(map(int, lines[i + 2].split()))
entropies(numLeavesSt, marks, childLeft, childRight)
print(prune_count_leaves(childLeft, childRight, numLeavesSt))

def findKClosest(array, k: int, value):
    array = [x - value for x in array]
    return array


array = [1, 3, 4, 5, 6, 6]
print(findKClosest(array, 3, 4))

import numpy as np

def find_pairs(maxA):
    for a in range(1, maxA):
        for b in range(1,a+1):
            numer = a*a + b*b
            denom = a*b + 1
            if(numer % denom == 0):
                print(a, b, numer//denom)

find_pairs(10000)

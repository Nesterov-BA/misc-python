from itertools import product

def calculate_probabilities(n):
    if n == 1:
        return (0.0, 1.0, 0.0)
    
    total = 0
    win_A = 0
    draw = 0
    win_B = 0
    
    for sequence in product(['O', 'R'], repeat=n):
        X = 0
        Y = 0
        for i in range(n - 1):
            if sequence[i] == 'O' and sequence[i + 1] == 'R':
                X += 1
            elif sequence[i] == 'O' and sequence[i + 1] == 'O':
                Y += 1
        
        if X > Y:
            win_A += 1
        elif X == Y:
            draw += 1
        else:
            win_B += 1
        total += 1
    
    prob_A = win_A / total
    prob_draw = draw / total
    prob_B = win_B / total
    
    return (prob_A, prob_draw, prob_B)

n = int(input())
prob_A, prob_draw, prob_B = calculate_probabilities(n)
print(f"{prob_A:.6f} {prob_draw:.6f} {prob_B:.6f}")

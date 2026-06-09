MOD = 998244353

def factorials(length, mod):
    fact = [1]*(length+1)
    for i in range(1, length+1):
        fact[i] = fact[i-1]* i %mod
    inv_fact = [1]*(length+1)
    inv_fact[length] = pow(fact[length], mod-2, mod)
    for i in range(length-1, -1, -1):
        inv_fact[i] = inv_fact[i+1]*(i+1)%mod
    return fact, inv_fact

def binom(N, k, fact, inv_fact, mod):
    if k < 0 or N < 0:
        return 0
    return (fact[N]*inv_fact[N-k]% mod)*inv_fact[k] % mod

def solve(N, W):
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for i in range(N-1):
        v1, v2 = list(map(int, input().split()))
        adj[v1].append(v2)
        adj[v2].append(v1)
        deg[v1]+=1
        deg[v2]+=1
    leaves = [v for v in range(1,N+1) if deg[v]==1]
    fact, inv_fact = factorials(N  + W + 1, MOD)
    if len(leaves) == 2:
        return binom(W + N-1, N-1, fact, inv_fact, MOD)
    if len(leaves) <= 1:
        return 0
    tails = []
    for leaf in leaves:
        prev, curr, length = -1, leaf, 0
        while True:
            nxt = next(v for v in adj[curr] if v != prev)
            prev, curr = curr, nxt
            length += 1
            if deg[curr] >= 3:
                tails.append(length)
                break
    answer = 0
    for k in range(W//2 + 1):
        product = 1
        for p in tails:
            product = product* binom(k + p - 1, p-1, fact, inv_fact, MOD) % MOD
        answer = (answer + product) % MOD
    return answer

N, W = list(map(int, input().split()))
print(solve(N,W))
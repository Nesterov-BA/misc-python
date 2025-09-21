def hanoi_visual(n, source, target, auxiliary, pegs=None):
    if pegs is None:
        pegs = {
            source: list(range(n, 0, -1)),
            auxiliary: [],
            target: []
        }

    if n > 0:
        hanoi_visual(n-1, source, auxiliary, target, pegs)

        disk = pegs[source].pop()
        pegs[target].append(disk)
        print(f"Move disk {disk} from {source} to {target}")
        print(f"Current state: {pegs}")

        hanoi_visual(n-1, auxiliary, target, source, pegs)

# Example usage
hanoi_visual(3, 'A', 'C', 'B')

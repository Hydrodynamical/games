import itertools
jump_directions = []
for perm in itertools.permutations([1, -1, 2, -2], 2):
    if (perm[0] + perm[1]) % 2 == 1:
        jump_directions.append(perm)

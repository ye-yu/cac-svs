def generate_uniques(*files):
    unique = set()
    for file in files:
        with open(file) as fio:
            lines = fio.readlines()
            lines = [line.strip().split() for line in lines]
            for line in lines:
                unique = unique.union(set(line))
    return unique

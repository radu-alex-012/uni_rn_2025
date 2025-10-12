import pathlib


# 1
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []
    variables = ("x", "y", "z")
    with open(path) as f:
        for line in f:
            tokens = line.split()
            row = []
            position = 0
            for var in variables:
                coefficient = 0.0
                sign = 1.0
                if tokens[position] in ("+", "-"):
                    sign = -1.0 if tokens[position] == "-" else 1.0
                    position += 1
                if var in tokens[position]:
                    token = tokens[position].strip(var)
                    match token:
                        case "-":
                            coefficient = -1.0
                        case "":
                            coefficient = 1.0
                        case _:
                            coefficient = float(token)
                    position += 1
                coefficient *= sign
                row.append(coefficient)
            A.append(row)
            B.append(float(tokens[-1]))
    return A, B


A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")


# 2
def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return (
        matrix[0][0] * matrix[1][1] * matrix[2][2]
        + matrix[0][1] * matrix[1][2] * matrix[2][0]
        + matrix[0][2] * matrix[1][0] * matrix[2][1]
        - matrix[0][2] * matrix[1][1] * matrix[2][0]
        - matrix[0][0] * matrix[1][2] * matrix[2][1]
        - matrix[0][1] * matrix[1][0] * matrix[2][2]
    )


print(f"{determinant(A)=}")


# 3
def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


print(f"{trace(A)=}")


# 4
def norm(vector: list[float]) -> float:
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


print(f"{norm(B)=}")


# 5
def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


print(f"{transpose(A)=}")


# 6
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(x * y for x, y in zip(row, vector)) for row in matrix]


print(f"{multiply(A, B)=}")


# 7
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    import copy

    det = determinant(matrix)
    result = []
    for col in range(3):
        mat = copy.deepcopy(matrix)
        for row in range(3):
            mat[row][col] = vector[row]
        result.append(determinant(mat) / det)
    return result


print(f"{solve_cramer(A, B)=}")


# 8
def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    import copy

    mat = copy.deepcopy(matrix)
    for row in range(3):
        del mat[row][j]
    del mat[i]
    return mat


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    mat = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append((-1) ** (i + j) * determinant(minor(matrix, i, j)))
        mat.append(row)
    return mat


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    scalar = 1 / determinant(matrix)
    inverse = [[element * scalar for element in row] for row in adjoint(matrix)]
    return multiply(inverse, vector)


print(f"{solve(A, B)=}")

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def a(x, y):
    z = 0
    while x > y:
        x /= 2
        y *= 2
        z += a(x, y)

    return z


def b(x, y, z):
    return a(x, y) + a(y, z) * a(z, x)


def c(x):
    if x < 0:
        return b(x - 1, x, x + 1)
    else:
        return 10


if __name__ == "__main__":
    n = 30
    for i in range(n):
        x = fib(i)
        x += a(x, i)
        x -= b(i, x, i + x)
        x *= c(i)

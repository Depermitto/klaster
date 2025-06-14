def process_strings(count):
    result = []
    for i in range(count):
        s = " " * i * 1000
        result.append(s)
    return result


if __name__ == "__main__":
    for _ in range(100):
        strings = process_strings(1000)

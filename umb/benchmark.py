from time import time

def benchmark(callback, *args, number_of_samples):
    results = []
    for i in range(number_of_samples):
        start = time()
        callback(*args)
        end = time()
        results.append(end - start)

    average = sum(results) / number_of_samples
    return average
import multiprocessing
import time
import matplotlib.pyplot as plt

def replace_pair(shared_ids, pair, symbol, empty_symbol=-1):
    """Replaces all instances of a given neighboring pair with [symbol, empty_symbol] in shared memory."""
    index = 0
    # Iterate over the shared memory list, checking for the pair
    while index < len(shared_ids) - 1:
        if shared_ids[index] == pair[0] and shared_ids[index+1] == pair[1]:
            shared_ids[index] = symbol         # Replace the first of the pair with the symbol
            shared_ids[index+1] = empty_symbol  # Replace the second with the empty symbol
            index += 2  # Skip the next element
            continue

        index += 1

    copy = shared_ids[:]

    return copy
        

def replace_multiple(pairs, shared_ids):
    """
    Replace multiple neighboring pairs in the shared 'ids' using multiprocessing.
    
    :param pairs_to_symbols: A list of tuples, where each tuple contains:
                              (pair, symbol, empty_symbol)
    :param shared_ids: Shared array for multiprocessing (no locking).
    """
    # Create a pool of processes
    processes = []
    for i, pair in enumerate(pairs):
        # Create a new process for each pair to replace
        p = multiprocessing.Process(target=replace_pair, args=(shared_ids, pair, 256+i))
        processes.append(p)
        p.start()
    
    # Ensure all processes complete
    for p in processes:
        p.join()

def replace_sequential(pairs, shared_ids):
    """
    Replace multiple neighboring pairs in the shared 'ids' sequentially.
    
    :param pairs_to_symbols: A list of tuples, where each tuple contains:
                              (pair, symbol, empty_symbol)
    :param shared_ids: Shared array for multiprocessing (no locking).
    """
    result = shared_ids[:]
    for i, pair in enumerate(pairs):
        result = replace_pair(result, pair, 256+i)

    return result

def benchmark(callback, *args):
    start = time.time()
    callback(*args)
    end = time.time()
    return end - start

def verify(a, b):
    for i, j in zip(a, b):
        if i != j:
            return False
        
    return True

def get_frequencies(data):
    freq = {}

    for i, j in zip(data, data[1:]):
        if (i, j) in freq:
            freq[(i, j)] += 1
        else:
            freq[(i, j)] = 1

    return freq

def benchmark_unit_test(data, num_processes):
    freq = get_frequencies(data)
    order_freq = sorted(freq, key=lambda x: freq[(x[0], x[1])], reverse=True)
    sample_pairs = []
    sample_numbers = []
    index = 0
    while len(sample_pairs) < num_processes or index < 55:
        if order_freq[index][0] not in sample_numbers and order_freq[index][1] not in sample_numbers and order_freq[index]:
            sample_pairs.append(order_freq[index])
            sample_numbers.append(order_freq[index][0])
            sample_numbers.append(order_freq[index][1])
        
        index += 1
    
    # Define the global list (shared memory) as a multiprocessing Array without a lock
    shared_data = multiprocessing.Array('i', data, lock=False)  # Shared array without locking
    time_a = benchmark(replace_multiple, sample_pairs, shared_data)
    time_b = benchmark(replace_sequential, sample_pairs, data)
    return time_b / time_a

if __name__ == "__main__":
    # Define the pairs and corresponding symbols to replace with
    with open("data/threebody.txt", "r") as f:
        text = f.read()

    # Tokenize the text into utf-8 bytes
    tokens = list(text.encode("utf-8"))
    results = []
    for i in range(1, 100):
        result = benchmark_unit_test(tokens, i)
        print(f"Speedup for {i} processes: {result}x")
        results.append(result)


    # Plot the results with matplotlib
    
    plt.plot(range(1, 100), results)
    plt.xlabel("Number of Processes")
    plt.ylabel("Speedup")
    plt.title("Speedup of multiprocessing vs sequential processing")
    plt.show()

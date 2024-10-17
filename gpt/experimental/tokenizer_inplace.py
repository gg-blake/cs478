freq0 = {}
def get_frequencies(ids):
    result = {}
    for p0, p1 in zip(ids, ids[1:]):
        if (p0, p1) not in result.keys():
            result[(p0, p1)] = 1
            continue

        result[(p0, p1)] = result[(p0, p1)] + 1

    return result

def increase_frequency(pair):
    if pair not in freq0.keys():
        freq0[pair] = 1
    else:
        freq0[pair] += 1

def decrease_frequency(pair):
    if pair not in freq0.keys():
        freq0[pair] = 0
    else:
        freq0[pair] -= 1
        if freq0[pair] <= 0:
            del freq0[pair] 

def merge(ids, pair, symbol):
    result = []
    freq0[(symbol, symbol)] = 0
    index = 0
    while index < len(ids):
        if index < len(ids) - 1 and ids[index] == pair[0] and ids[index+1] == pair[1]:
            result.append(symbol)
            
            if index - 2 >= 0:
                LNPO = (ids[index-1], ids[index]) # Left Neighbor Pair Old
                LRP = (ids[index-1], symbol)
                decrease_frequency(LNPO)
                # Check if there was a matching pair immediately before
                if ids[index-2] == pair[0] and ids[index-1] == pair:
                    # If there isn't an immediate neighboring pair,
                    decrease_frequency(LNPO) # This will prevent a redundant increase of the pair (ids[index-1], ids[index])
                    decrease_frequency((symbol, ids[index])) # We need to undo a frequency update of the last pair (symbol, ids[index])
                else:
                    # If there isn't an immediate neighboring pair, then we can increment the replacement left pair by one
                    increase_frequency(LRP)
            elif index - 1 >= 0:
                # Check if there was a matching pair immediately before
                LNPO = (ids[index-1], ids[index]) # Left Neighbor Pair Old
                decrease_frequency(LNPO)
                LRP = (ids[index-1], symbol)
                increase_frequency(LRP)

            if index + 3 < len(ids):
                RNPO = (ids[index+1], ids[index+2]) # Right Neighbor Pair Old
                RRP = (symbol, ids[index+2]) # Right Replacement Pair
                decrease_frequency(RNPO)
                if  not (ids[index+2] == pair[0] and ids[index+3] == pair[1]):
                    increase_frequency(RRP)
                else:
                    increase_frequency((symbol, symbol))
            elif index + 2 < len(ids):
                RNPO = (ids[index+1], ids[index+2]) # Right Neighbor Pair Old
                RRP = (symbol, ids[index+2]) # Right Replacement Pair
                decrease_frequency(RNPO)
                increase_frequency(RRP)

            index += 2

        else:
            result.append(ids[index])
            index += 1

    if not freq0[pair] > 0:
        del freq0[pair]

    return result

def bpe(ids, num_iterations):
    result = ids[:]
    freq = get_frequencies(ids)
    for i in range(num_iterations):
        freq_pair = max(freq, key=lambda x:freq[(x[0], x[1])])
        merge(result, )


if __name__ == "__main__":
    ids0 = [21, 45, 20, 21, 45, 21, 45, 10, 21, 45, 10, 21, 21, 45, 11, 10, 21, 45, 21, 45, 21, 45, 38]
    ids1 = ids0[:]
    freq0 = get_frequencies(ids0)
    #print(freq)
    ids0 = merge(ids0, (21, 45), 99)
    freq1 = get_frequencies(ids1)
    freq2 = get_frequencies(ids0)

    print(f"IDS before merge(): {ids1}\nFreq aefore merge(): {freq1}\nIDS after merge(): {ids0}\nFreq modified by merge(): {freq0}\nFreq after calculated by get_frequencies(): {freq2}")
    for i in freq2.keys():
        print(f"Verify {i}:{freq2[i]}, Test {i}:{freq0[i]}")
    
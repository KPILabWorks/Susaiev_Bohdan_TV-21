def multi_map(iterable, *funcs):
    for item in iterable:
        for func in funcs:
            item = func(item)
        yield item

data = [32, 43, 12, 7]

def add_one(x): return x + 1
def square(x): return x * x

result = list(multi_map(data, add_one, square))
print(result) 

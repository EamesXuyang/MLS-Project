import numpy as np

class A:
    def __init__(self):
        self.data = [1, 2, 3, 4]
    
    def __getitem__(self, item):
        print(item)
        print(np.array(item))
        return self.data[item]


a = A()
print(a[1:3])
print(a[1:3:2])
print(a[1:3:1])
print(a[1:3:0])
print(a[1:3:-1])

print(isinstance(range(3), slice))
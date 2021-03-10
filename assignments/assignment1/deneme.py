import numpy as np
num_folds = 5
train_folds_X = []
train_folds_y = []

# a=np.arange(5)
a=list(range(5))
# print(a)

# desired output   1234 0234 0134 0124 0123

# for i in range(len(a)):
#     l1 = a[0:i]
#     a[i + 1 :]
#     print(a[0:i], a[i+1:] )

for i in range(len(a)):
        l1 = a[0:i]
        l2 = a[i + 1:]
        l1.extend(l2)
        print(l1)
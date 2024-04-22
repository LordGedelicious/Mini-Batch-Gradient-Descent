import csv
import numpy as np

def parse_csv(file):
    data = []
    with open(file, 'r') as iris:
        read = csv.reader(iris)
        next(read)
        for row in read:
            ft = np.array(row[0:5], dtype = float)
            name = row[5]
            if (name == 'Iris-setosa'):
                label = np.array([1,0,0])
            elif (name == 'Iris-versicolor'):
                label = np.array([0,1,0])
            elif (name == 'Iris-virginica'):
                label = np.array([0,0,1])
            data.append((ft, label))
    return data

file = 'testcase/iris.csv'
parse = parse_csv(file)
print(parse)
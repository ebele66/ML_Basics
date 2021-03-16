import matplotlib.pyplot as plt
import numpy as np


def preprocess(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            row = line.strip().split()
            data.append(row)
        data = np.array(data, dtype=np.float)
    return data


# plot of data points and the cluster centers
def plot(trace):
    colmap = {0: 'r', 1: 'g', 2: 'b', 3: 'm', 4: 'y', 5: 'c', 6: 'k', 7: 'w'}
    iter = 0
    for centers, pts in trace:
        fig = plt.figure(figsize=(4, 4))
        for i in pts.keys():
            row = pts[i]
            for a in row:
                plt.scatter(*a, color=colmap[i])
        for c in centers:
            plt.scatter(*c, marker="+")
        plt.savefig("iter" + str(iter))
        iter += 1


# Euclidean distance between two points
def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# assign data pointers into K clusters
def assign(k, M, points):
    res = {p: [] for p in range(k)}
    for x in points:
        row = []
        for m in M:
            row.append(distance(x, m) ** 2)
        act = np.argmin(row)   # choosing the cluster with the smallest distance between point and cluster center
        res[act].append(x)
    return res


# recalculate the center of the clusters
def updateM(res):
    M = []
    for i in res.keys():
        M.append(np.mean(np.array(res[i]), dtype=np.float64, axis=0))
    return np.array(M)


def converged(old, new):
    return np.all(old == new)


def kmeans(k, r, points):
    squares = []
    trace = {rr: [] for rr in range(r)}   # iteration trace for each initialization
    for rr in range(r):
        init_mean = np.random.choice(1500, k, replace=False)
        M = np.array([points[x] for x in init_mean])
        while True:
            prev = M
            res = assign(k, M, points)
            trace[rr].append((M, res))
            M = updateM(res)
            if converged(prev, M):
                break
        sse = 0   # calculate sum of square error for each initialization
        for key in res.keys():
            for x in res[key]:
                sse += (distance(x, M[key]) ** 2)
        squares.append(sse)
    soln = np.argmin(squares)
    return squares[soln], trace[soln]


points = preprocess("GMM_data_fall2019.txt")
ks = [2, 5, 7]
rs = [10, 10, 5]

for i in range(len(ks)):
    sse, trace = kmeans(ks[i], rs[i], points)
    plot(trace)
    print(sse)



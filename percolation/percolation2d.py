import random
from collections import deque
import numpy as np 

"""
Site percolation in 2d.  
"""

random.seed(12)

n = 100

di = [-1, 1, 0, 0]
dj = [0, 0, -1, 1]

def existsPath2d(p):
	a = [[0 for j in range(n)] for i in range(n)]

	for i in range(n):
		for j in range(n):
			r = random.random()
			if r < p:
				a[i][j] = 1

	b = [[0 for j in range(n)] for i in range(n)]

	q = deque()
	visited = set()
	for i in range(n):
		if a[i][n-1] == 1:
			q.append((i, n-1))
			visited.add((i, n-1))

	while len(q):
		i,j = q.popleft()
		b[i][j] = 1
		for m in range(len(di)):
			ii = i + di[m]
			jj = j + dj[m]
			if 0 <= ii < n and 0 <= jj < n:
				if (ii, jj) not in visited:
					if a[ii][jj] == 1:
						visited.add((ii,jj))
						q.append((ii,jj))


	for i in range(n):
		if b[i][0] == 1:
			return True
	return False


ntries = 100
num = 101
t = []
ps = np.linspace(0,1,num=num)
for m in range(num):
	p = ps[m]
	res = 0
	for _ in range(ntries):
		res += existsPath2d(p)
	res /= ntries
	print(f'{m=} {p=:.2f} {res=}')
	t.append(int(res))



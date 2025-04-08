import random
from collections import deque
import numpy as np 

"""
Site percolation in 3d.  
"""

random.seed(12)

n = 100

di = [-1, 1, 0, 0, 0, 0]
dj = [0, 0, -1, 1, 0, 0]
dk = [0, 0, 0, 0, -1, 1]

def existsPath3d(p):
	a = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]

	for i in range(n):
		for j in range(n):
			for k in range(n):
				r = random.random()
				if r < p:
					a[i][j][k] = 1

	b = [[[0 for k in range(n)] for j in range(n)] for i in range(n)]

	q = deque()
	visited = set()
	for i in range(n):
		for j in range(n):
			if a[i][j][n-1] == 1:
				q.append((i, j, n-1))
				visited.add((i, j, n-1))

	while len(q):
		i,j,k = q.popleft()
		b[i][j][k] = 1
		for m in range(len(di)):
			ii = i + di[m]
			jj = j + dj[m]
			kk = k + dk[m]
			if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
				if (ii, jj, kk) not in visited:
					if a[ii][jj][kk] == 1:
						visited.add((ii,jj,kk))
						q.append((ii,jj,kk))


	for i in range(n):
		for j in range(n):
			if b[i][j][0] == 1:
				return True
	return False


num = 101
t = []
ps = np.linspace(0,1,num=num)
for m in range(num):
	p = ps[m]
	res = existsPath3d(p)
	print(f'{m=} {p=:.2f} {res=}')
	t.append(int(res))



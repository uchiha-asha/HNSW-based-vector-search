import sys

l1, l2 = [], []

with open(sys.argv[1]) as file:
	l1 = [list(map(int, line.split())) for line in file]

with open(sys.argv[2]) as file:
	l2 = [list(map(int, line.split())) for line in file]

precision, recall = 0, 0

print(len(l1), len(l2))

for i in range(0, len(l1)):
	tmp = len(list(set(l1[i]) & set(l2[i])))
	precision = precision + tmp/len(l1[i])
	recall = recall + tmp/len(l2[i])

print(precision, recall)

precision = precision/len(l1)
recall = recall/len(l2)

print(precision, recall)
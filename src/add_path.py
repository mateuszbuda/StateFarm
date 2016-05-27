import sys

filename = sys.argv[1]
path = sys.argv[2]

if path[-1] != '/':
	path += '/'

fIn = open(filename, 'r')
fOut = open(filename.replace('.', '_path.'), 'w')

count = 0
prevC = "fsdfs"

# FIXME
for line in fIn:
	count += 1
	c = line[-2]
	if c != prevC:
		prevC = c
		count = 0

	if count > 1:
		continue

	l = path + 'c' + str(c) + '/' + line
	fOut.write(l)

fOut.close()
fIn.close()
import sys

filename = sys.argv[1]
path = sys.argv[2]

if path[-1] != '/':
	path += '/'

fIn = open(filename, 'r')
fOut = open(filename.replace('.', '_path.'), 'w')

for line in fIn:
	c = line[-2]
	l = path + 'c' + str(c) + '/' + line
	fOut.write(l)

fOut.close()
fIn.close()
.PHONY: t

all:

t:
	go test -v . ./nd ./matrix

test_mnist: test_data
	go test -v ./batch ./single

test: t test_mnist

test_data: train-images-idx3-ubyte.idx train-labels-idx1-ubyte.idx t10k-images-idx3-ubyte.idx t10k-labels-idx1-ubyte.idx

%.idx : %.gz
	gunzip -c $< > $@

train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz:
	curl -o $@ http://yann.lecun.com/exdb/mnist/$@


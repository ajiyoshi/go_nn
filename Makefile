all:

test: test_data
	go test -v
	cd batch && go test -v
	cd single && go test -v
	cd matrix && go test -v
	cd mnist && go test -v

test_data: train-images-idx3-ubyte.idx train-labels-idx1-ubyte.idx t10k-images-idx3-ubyte.idx t10k-labels-idx1-ubyte.idx

%.idx : %.gz
	gunzip -c $< > $@

train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz:
	curl -o $@ http://yann.lecun.com/exdb/mnist/$@


all: run

hoge:
	docker run --rm -it -v `pwd`:/var/python np-dump python /var/python/hoge.py

dump: prepare
	docker run --rm -it \
		-v `pwd`:/var/python \
		-w /var/python/deep-learning-from-scratch/ch07 \
		np-dump \
		python dump.py
	mv deep-learning-from-scratch/ch07/*.mp ../../t/

run: dump
	go run main.go

pre_run: build
	docker run --rm -it -v `pwd`:/var/python np-dump python /var/python/hoge.py

build:
	docker build -t np-dump docker/

prepare:
	cp ./dump.py deep-learning-from-scratch/ch07

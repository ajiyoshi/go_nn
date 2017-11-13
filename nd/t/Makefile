all: dump

dump: build
	docker run --rm -it -v `pwd`:/var/python np-dump python /var/python/dump.py

build:
	docker build -t np-dump .

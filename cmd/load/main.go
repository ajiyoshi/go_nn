package main

import (
	"fmt"
	"github.com/vmihailenco/msgpack"
	"os"
)

func main() {
	err := run()
	if err != nil {
		panic(err)
	}
	fmt.Println("vim-go")
}

type npArrayDump struct {
	Type  string `msgpack:"type"`
	Data  []byte `msgpack:"data"`
	Shape []int  `msgpack:"shape"`
}

func run() error {
	f, err := os.Open("deep-learning-from-scratch/ch07/colW")
	if err != nil {
		return err
	}

	d := msgpack.NewDecoder(f)
	buf := &npArrayDump{}
	if err := d.Decode(buf); err != nil {
		return err
	}

	fmt.Println(buf.Shape)
	return nil
}

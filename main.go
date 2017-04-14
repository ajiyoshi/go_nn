package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"os"
)

func main() {
	err := Main2()
	if err != nil {
		fmt.Println(err)
	}
}

func Main2() error {
	x := mat64.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	w := mat64.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	var ret mat64.Dense
	ret.Mul(x, w)

	fmt.Printf("%v\n",
		mat64.Formatted(&ret, mat64.Prefix(" "), mat64.Excerpt(3)))
	return nil
}

func Main() error {
	m, err := NewMnist("./train-images-idx3-ubyte", "./train-labels-idx1-ubyte")
	if err != nil {
		return err
	}
	defer m.Close()

	fmt.Printf("len:%d, rows:%d, cols:%d\n",
		m.Images.Num, m.Images.Rows, m.Images.Cols)

	for i := 0; i < 10; i++ {
		err := m.Jump(i)
		if err != nil {
			return err
		}

		err = Write(fmt.Sprintf("hoge%d-%d.png", i, m.Label()), m)
		if err != nil {
			return err
		}
	}

	return nil
}

func Write(path string, i *Mnist) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return i.DumpPng(f)
}

func or(x1, x2 int) int {
	x := mat64.NewVector(2, []float64{float64(x1), float64(x2)})
	w := mat64.NewVector(2, []float64{0.5, 0.5})
	b := -0.2

	v := mat64.Dot(x, w) + b
	if v <= 0 {
		return 0
	} else {
		return 1
	}
}

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ajiyoshi/gocnn"
	"github.com/ajiyoshi/gocnn/mnist"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	err := run()
	if err != nil {
		panic(err)
	}
}

func run() error {
	m, err := mnist.NewMnist("../../train-images-idx3-ubyte.idx", "../../train-labels-idx1-ubyte.idx")
	if err != nil {
		return err
	}
	defer m.Close()

	len := m.Images.Rows * m.Images.Cols
	N := 25
	shape := gocnn.NewShape(N, 1, m.Images.Rows, m.Images.Cols)

	cnn := gocnn.NewSimpleConvNet()
	buf := mnist.NewTrainBuffer(N, len, 10)
	for i := 0; i < 100; i++ {
		index := rand.Intn(m.Images.Num - N)
		at := mnist.Seq(index, N)
		buf.Load(m, at)
		x, t := buf.Bake()
		img := gocnn.NewReshaped(shape, x)
		loss := cnn.Train(img, t)
		fmt.Printf("%f, %f\n", loss, cnn.Accracy(img, t))
	}

	return nil
}

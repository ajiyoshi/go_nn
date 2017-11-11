package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/ajiyoshi/gocnn"
	"github.com/ajiyoshi/gocnn/mnist"
)

func init() {
	rand.Seed(time.Now().Unix())
}

func main() {
	cpuprofile := "mycpu.prof"
	f, err := os.Create(cpuprofile)
	if err != nil {
		panic(err)
	}
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	err = run()
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
	for i := 0; i < 1; i++ {
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

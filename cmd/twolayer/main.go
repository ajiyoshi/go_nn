package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ajiyoshi/gocnn/batch"
	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/mnist"
	"github.com/ajiyoshi/gocnn/optimizer"
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
	rows := 25
	hidden := 50
	output := 10
	optimizer := optimizer.NewMomentumFactory(0.1, 0.1)
	layer := NewTwoLayerNN(len, hidden, output, optimizer)
	nn := batch.NewNeuralNet(layer)

	buf := mnist.NewTrainBuffer(rows, len, 10)
	for i := 0; i < 100; i++ {
		index := rand.Intn(m.Images.Num - rows)
		at := mnist.Seq(index, rows)
		buf.Load(m, at)
		x, t := buf.Bake()
		fmt.Printf("x(%d) %s\n", index, matrix.Summary(x))
		fmt.Printf("W(%d) %s\n", index, matrix.Summary(layer.affine1.Weight))
		fmt.Printf("dW(%d) %s\n", index, matrix.Summary(layer.affine1.DWeight))
		loss := nn.Train(x, t)
		fmt.Println(loss)
	}

	return nil
}

type TwoLayerNN struct {
	affine1 *batch.AffineLayer
	leru1   *batch.ReLULayer
	affine2 *batch.AffineLayer
	leru2   *batch.ReLULayer
	last    *batch.SoftMaxWithLoss
}

const WeightInitStd = 0.1

func NewTwoLayerNN(input_size, hidden_size, output_size int, f optimizer.OptimizerFactory) *TwoLayerNN {
	return &TwoLayerNN{
		affine1: batch.NewAffine(WeightInitStd, input_size, hidden_size, f()),
		leru1:   batch.NewReLU(),
		affine2: batch.NewAffine(WeightInitStd, hidden_size, output_size, f()),
		leru2:   batch.NewReLU(),
		last:    batch.NewSoftMaxWithLoss(),
	}
}

func (nn *TwoLayerNN) Layers() []batch.Layer {
	return []batch.Layer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2,
	}
}
func (nn *TwoLayerNN) Last() batch.LastLayer {
	return nn.last
}

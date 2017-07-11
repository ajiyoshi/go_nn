package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/ajiyoshi/gocnn"
	"github.com/ajiyoshi/gocnn/batch"
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

	input := m.Images.Rows * m.Images.Cols
	hidden := 100
	output := 10
	optimizer := gocnn.NewMomentumFactory(0.1, 0.1)

	layer := NewFiveLayerNN(input, hidden, output, optimizer)
	nn := batch.NewNeuralNet(layer)

	batchSize := 200
	buf := mnist.NewTrainBuffer(batchSize, input, 10)
	for i := 0; i < 3000; i++ {
		at := mnist.RandamSeq(batchSize, m.Images.Num)
		buf.Load(m, at)
		x, t := buf.Bake()
		loss := nn.Train(x, t)
		if i%100 == 0 {
			fmt.Printf("%f, %f\n", loss, nn.Accracy(x, t))
		}
	}

	m2, err := mnist.NewMnist("../../t10k-images-idx3-ubyte.idx", "../../t10k-labels-idx1-ubyte.idx")
	if err != nil {
		return err
	}
	defer m2.Close()

	buf = mnist.NewTrainBuffer(10000, input, 10)
	buf.Load(m2, mnist.Seq(0, 10000))
	x, t := buf.Bake()
	fmt.Printf("test:%f, %f\n", nn.Loss(x, t), nn.Accracy(x, t))

	return nil
}

type FiveLayerNN struct {
	affine1 *batch.AffineLayer
	leru1   *batch.ReLULayer
	affine2 *batch.AffineLayer
	leru2   *batch.ReLULayer
	affine3 *batch.AffineLayer
	leru3   *batch.ReLULayer
	affine4 *batch.AffineLayer
	leru4   *batch.ReLULayer
	last    *batch.SoftMaxWithLoss
}

const WeightInitStd = 0.1

var _ batch.NeuralNetLayers = (*FiveLayerNN)(nil)

func NewFiveLayerNN(input_size, hidden_size, output_size int, f gocnn.OptimizerFactory) *FiveLayerNN {
	return &FiveLayerNN{
		affine1: batch.NewAffine(WeightInitStd, input_size, hidden_size, f()),
		leru1:   batch.NewReLU(),
		affine2: batch.NewAffine(WeightInitStd, hidden_size, hidden_size, f()),
		leru2:   batch.NewReLU(),
		affine3: batch.NewAffine(WeightInitStd, hidden_size, hidden_size, f()),
		leru3:   batch.NewReLU(),
		affine4: batch.NewAffine(WeightInitStd, hidden_size, output_size, f()),
		leru4:   batch.NewReLU(),
		last:    batch.NewSoftMaxWithLoss(),
	}
}

func (nn *FiveLayerNN) Layers() []batch.Layer {
	return []batch.Layer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2, nn.affine3, nn.leru3, nn.affine4, nn.leru4,
	}
}
func (nn *FiveLayerNN) Last() batch.LastLayer {
	return nn.last
}

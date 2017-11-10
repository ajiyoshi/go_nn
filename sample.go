package gocnn

import (
	"github.com/ajiyoshi/gocnn/batch"
	"github.com/ajiyoshi/gocnn/optimizer"
)

const WeightInitStd = 0.1

func NewSimpleConvNet() *SimpleCNN {
	opt := optimizer.NewMomentumFactory(0.1, 0.1)
	nnParam := &batch.NNParam{
		InputSize:  4320,
		HiddenSize: 100,
		OutputSize: 10,
	}
	nnLayer := batch.New2LayerNN(nnParam, opt)

	cnnParam := &CNNParam{
		FilterNum:  30,
		Channel:    1,
		FilterSize: 5,
		Stride:     1,
		Pad:        0,
	}
	cnn := NewSingleCNN(cnnParam, opt)

	return &SimpleCNN{
		imageLayers: []ImageLayer{cnn.Conv, cnn.Relu, cnn.Pool},
		nn:          batch.NewNeuralNet(nnLayer),
	}
}

type SingleCNN struct {
	Conv *Convolution
	Relu *ReLU
	Pool *Pooling
}

type CNNParam struct {
	FilterNum  int
	FilterSize int
	Channel    int
	Stride     int
	Pad        int
}

func NewSingleCNN(conf *CNNParam, f optimizer.OptimizerFactory) *SingleCNN {
	shape := &Shape{
		N:   conf.FilterNum,
		Ch:  conf.Channel,
		Row: conf.FilterSize,
		Col: conf.FilterSize,
	}
	return &SingleCNN{
		Conv: NewConvolution(shape, conf.Stride, conf.Pad, f()),
		Relu: &ReLU{},
		Pool: &Pooling{
			Row:    2,
			Col:    2,
			Stride: 2,
		},
	}
}

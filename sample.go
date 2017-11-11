package gocnn

import (
	"fmt"
	"github.com/ajiyoshi/gocnn/batch"
	"github.com/ajiyoshi/gocnn/optimizer"
)

const WeightInitStd = 0.01

func NewSimpleConvNet() *SimpleCNN {
	opt := optimizer.NewMomentumFactory(0.1, 0.1)

	cnnParam := &CNNParam{
		FilterNum:  30,
		Channel:    1,
		FilterSize: 5,
		Stride:     1,
		Pad:        0,
	}
	cnn := NewSingleCNN(cnnParam, opt)

	//input_size = input_dim[1]
	//conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
	//pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
	convOutput := (28-cnnParam.FilterSize+2*cnnParam.Pad)/cnnParam.Stride + 1
	poolOutput := cnnParam.FilterNum * (convOutput / 2) * (convOutput * 2)
	fmt.Println(poolOutput)

	nnParam := &batch.NNParam{
		InputSize:  4320,
		HiddenSize: 100,
		OutputSize: 10,
	}
	nnLayer := batch.New2LayerNN(nnParam, opt)

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

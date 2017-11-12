package gocnn

import (
	"github.com/ajiyoshi/gocnn/batch"
	"github.com/ajiyoshi/gocnn/optimizer"
)

const WeightInitStd = 0.01

func NewSimpleConvNet(s *Shape) *SimpleCNN {
	//opt := optimizer.NewMomentumFactory(0.1, 0.1)
	opt := optimizer.NewAdam(0.001, 0.9, 0.999)

	cp := &CNNParam{
		FilterNum:  30,
		Channel:    s.Ch,
		FilterSize: 5,
		Stride:     1,
		Pad:        0,
	}
	cnn := NewSingleCNN(cp, opt)

	//input_size = input_dim[1]
	//conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
	//pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
	convOutRow := (s.Row+2*cp.Pad-cp.FilterSize)/cp.Stride + 1
	convOutCol := (s.Col+2*cp.Pad-cp.FilterSize)/cp.Stride + 1
	poolOutput := cp.FilterNum * (convOutRow / 2) * (convOutCol / 2)

	nnParam := &batch.NNParam{
		InputSize:  poolOutput,
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

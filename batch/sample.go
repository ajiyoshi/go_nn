package batch

import (
	"github.com/ajiyoshi/gocnn/optimizer"
)

type NNParam struct {
	InputSize  int
	HiddenSize int
	OutputSize int
}

type TwoLayerNN struct {
	Affine1 *AffineLayer
	Relu1   *ReLULayer
	Affine2 *AffineLayer
	Relu2   *ReLULayer
	SoftMax *SoftMaxWithLoss
}

const WeightInitStd = 0.01

func New2LayerNN(param *NNParam, f optimizer.OptimizerFactory) *TwoLayerNN {
	return &TwoLayerNN{
		Affine1: NewAffine(WeightInitStd, param.InputSize, param.HiddenSize, f()),
		Relu1:   NewReLU(),
		Affine2: NewAffine(WeightInitStd, param.HiddenSize, param.OutputSize, f()),
		Relu2:   NewReLU(),
		SoftMax: NewSoftMaxWithLoss(),
	}
}
func NewTwoLayerNN(input_size, hidden_size, output_size int, f optimizer.OptimizerFactory) *TwoLayerNN {
	return &TwoLayerNN{
		Affine1: NewAffine(WeightInitStd, input_size, hidden_size, f()),
		Relu1:   NewReLU(),
		Affine2: NewAffine(WeightInitStd, hidden_size, output_size, f()),
		Relu2:   NewReLU(),
		SoftMax: NewSoftMaxWithLoss(),
	}
}

func (nn *TwoLayerNN) Layers() []Layer {
	return []Layer{
		nn.Affine1, nn.Relu1, nn.Affine2, nn.Relu2,
	}
}
func (nn *TwoLayerNN) Last() LastLayer {
	return nn.SoftMax
}

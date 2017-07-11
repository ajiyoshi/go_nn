package single

import (
	"github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/optimizer"
)

type TwoLayerNN struct {
	affine1 *AffineLayer
	leru1   *ReLULayer
	affine2 *AffineLayer
	leru2   *ReLULayer
	last    *SoftMaxWithLoss
}

func NewTwoLayerNN(input_size, hidden_size, output_size int, f optimizer.OptimizerFactory) *TwoLayerNN {
	return &TwoLayerNN{
		affine1: NewAffine(input_size, hidden_size, f()),
		leru1:   &ReLULayer{},
		affine2: NewAffine(hidden_size, output_size, f()),
		leru2:   &ReLULayer{},
		last:    &SoftMaxWithLoss{},
	}
}

func (nn *TwoLayerNN) Layers() []Layer {
	return []Layer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2,
	}
}
func (nn *TwoLayerNN) Last() LastLayer {
	return nn.last
}

const WeightInitStd = 0.1
const weightInitStd = WeightInitStd

func NewAffine(input, output int, op optimizer.Optimizer) *AffineLayer {
	w := matrix.RandamDense(input, output)
	w.Scale(weightInitStd, w)
	b := mat64.NewVector(output, nil)
	return NewAffineLayer(w, b, op)
}

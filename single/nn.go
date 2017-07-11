package single

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"

	"github.com/ajiyoshi/gocnn"
)

type TwoLayerNN struct {
	affine1 *AffineLayer
	leru1   *ReLULayer
	affine2 *AffineLayer
	leru2   *ReLULayer
	last    *SoftMaxWithLoss
}

func NewTwoLayerNN(input_size, hidden_size, output_size int, f gocnn.OptimizerFactory) *TwoLayerNN {
	return &TwoLayerNN{
		affine1: InitAffineLayer(input_size, hidden_size, f()),
		leru1:   &ReLULayer{},
		affine2: InitAffineLayer(hidden_size, output_size, f()),
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

func InitAffineLayer(input, output int, op gocnn.Optimizer) *AffineLayer {
	w := RandamDense(input, output)
	w.Scale(weightInitStd, w)
	b := mat64.NewVector(output, nil)
	return NewAffineLayer(w, b, op)
}

func RandamDense(raws, cols int) *mat64.Dense {
	ret := mat64.NewDense(raws, cols, nil)
	for r := 0; r < raws; r++ {
		for c := 0; c < cols; c++ {
			ret.Set(r, c, rand.NormFloat64())
		}
	}
	return ret
}

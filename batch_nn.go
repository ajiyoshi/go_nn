package gocnn

import (
	"github.com/gonum/matrix/mat64"
)

type TwoLayerBatchNN struct {
	affine1 *BatchAffineLayer
	leru1   *BatchReLULayer
	affine2 *BatchAffineLayer
	leru2   *BatchReLULayer
	last    *BatchSoftMaxWithLoss
}

func NewTwoLayerBatchNN(input_size, hidden_size, output_size int, f OptimizerFactory) *TwoLayerBatchNN {
	return &TwoLayerBatchNN{
		affine1: InitAffineLayerB(input_size, hidden_size, f()),
		leru1:   &BatchReLULayer{},
		affine2: InitAffineLayerB(hidden_size, output_size, f()),
		leru2:   &BatchReLULayer{},
		last:    &BatchSoftMaxWithLoss{},
	}
}

func (nn *TwoLayerBatchNN) Layers() []BatchLayer {
	return []BatchLayer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2,
	}
}
func (nn *TwoLayerBatchNN) Last() BatchLastLayer {
	return nn.last
}

func InitAffineLayerB(input, output int, op Optimizer) *BatchAffineLayer {
	w := RandamDense(input, output)
	w.Scale(weightInitStd, w)
	b := mat64.NewVector(output, nil)
	return NewBatchAffineLayer(w, b, op)
}

type FiveLayerBatchNN struct {
	affine1 *BatchAffineLayer
	leru1   *BatchReLULayer
	affine2 *BatchAffineLayer
	leru2   *BatchReLULayer
	affine3 *BatchAffineLayer
	leru3   *BatchReLULayer
	affine4 *BatchAffineLayer
	leru4   *BatchReLULayer
	last    *BatchSoftMaxWithLoss
}

func NewFiveLayerBatchNN(input_size, hidden_size, output_size int, f OptimizerFactory) *FiveLayerBatchNN {
	return &FiveLayerBatchNN{
		affine1: InitAffineLayerB(input_size, hidden_size, f()),
		leru1:   &BatchReLULayer{},
		affine2: InitAffineLayerB(hidden_size, hidden_size, f()),
		leru2:   &BatchReLULayer{},
		affine3: InitAffineLayerB(hidden_size, hidden_size, f()),
		leru3:   &BatchReLULayer{},
		affine4: InitAffineLayerB(hidden_size, output_size, f()),
		leru4:   &BatchReLULayer{},
		last:    &BatchSoftMaxWithLoss{},
	}
}

func (nn *FiveLayerBatchNN) Layers() []BatchLayer {
	return []BatchLayer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2, nn.affine3, nn.leru3, nn.affine4, nn.leru4,
	}
}
func (nn *FiveLayerBatchNN) Last() BatchLastLayer {
	return nn.last
}

package main

import (
	"github.com/gonum/matrix/mat64"
	"math/rand"
)

type NN struct {
	affine1 *AffineLayer
	leru1   *ReLULayer
	affine2 *AffineLayer
	leru2   *ReLULayer
	last    *SoftMaxWithLoss
}

func NewNN(input_size, hidden_size, output_size int, f OptimizerFactory) *NN {
	return &NN{
		affine1: InitAffineLayer(input_size, hidden_size, f()),
		leru1:   &ReLULayer{},
		affine2: InitAffineLayer(hidden_size, output_size, f()),
		leru2:   &ReLULayer{},
		last:    &SoftMaxWithLoss{},
	}
}

func (nn *NN) Layers() []Layer {
	return []Layer{
		nn.affine1, nn.leru1, nn.affine2, nn.leru2,
	}
}
func (nn *NN) Reverse() []Layer {
	l := nn.Layers()
	len := len(l)
	ret := make([]Layer, len)
	for i, x := range l {
		ret[len-i-1] = x
	}
	return ret
}
func (nn *NN) Last() *SoftMaxWithLoss {
	return nn.last
}
func (nn *NN) Predict(x *mat64.Vector) *mat64.Vector {
	for _, layer := range nn.Layers() {
		x = layer.Forward(x)
	}
	return x
}
func (nn *NN) Loss(x, t *mat64.Vector) float64 {
	y := nn.Predict(x)
	return nn.Last().Forward(y, t)
}
func (nn *NN) BackProp() *mat64.Vector {
	dout := nn.Last().Backward(1)
	for _, layer := range nn.Reverse() {
		dout = layer.Backward(dout)
	}
	return dout
}
func (nn *NN) Update() {
	for _, layer := range nn.Layers() {
		layer.Update()
	}
}
func (nn *NN) Grad(x, t *mat64.Vector) []mat64.Matrix {
	nn.Loss(x, t)
	nn.BackProp()

	return []mat64.Matrix{
		nn.affine1.DWeight,
		nn.affine1.DBias,
		nn.affine2.DWeight,
		nn.affine2.DBias,
	}
}
func (nn *NN) NumGrad(x, t *mat64.Vector) []mat64.Matrix {
	f := func(m *mat64.Dense) float64 {
		return nn.Loss(x, t)
	}
	g := func(v *mat64.Vector) float64 {
		return nn.Loss(x, t)
	}

	return []mat64.Matrix{
		NumericalGradM(f, nn.affine1.Weight),
		NumericalGrad(g, nn.affine1.Bias),
		NumericalGradM(f, nn.affine2.Weight),
		NumericalGrad(g, nn.affine2.Bias),
	}
}

const weightInitStd = 0.01

func InitAffineLayer(input, output int, op Optimizer) *AffineLayer {
	w := RandamDense(input, output)
	w.Scale(weightInitStd, w)
	b := mat64.NewVector(output, nil)
	return NewAffineLayer(w, b, op)
}

func RandamDense(raws, cols int) *mat64.Dense {
	ret := mat64.NewDense(raws, cols, nil)
	for r := 0; r < raws; r++ {
		for c := 0; c < cols; c++ {
			ret.Set(r, c, rand.Float64())
		}
	}
	return ret
}

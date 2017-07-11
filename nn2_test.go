package gocnn

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

type Simple2NN struct {
	affine *AffineLayer
	relu   *ReLULayer
	last   *SoftMaxWithLoss
}

func NewSimple2NN(w *mat64.Dense) *Simple2NN {
	_, c := w.Dims()
	b := mat64.NewVector(c, nil)
	al := NewAffineLayer(w, b, NewMomentum(0.1, 0.1))
	return &Simple2NN{
		affine: al,
		relu:   &ReLULayer{},
		last:   &SoftMaxWithLoss{},
	}
}

var _ NNImpl = &Simple2NN{}

func (nn *Simple2NN) Layers() []Layer {
	return []Layer{nn.affine, nn.relu}
}
func (nn *Simple2NN) Last() LastLayer {
	return nn.last
}

func TestSimple2(t *testing.T) {
	for _, c := range []struct {
		title string
		W     *mat64.Dense
		x     *mat64.Vector
		t     *mat64.Vector
		dW    mat64.Matrix
	}{
		{
			title: "TestNNGrad",
			W: mat64.NewDense(2, 3, []float64{
				0.47355232, 0.9977393, 0.84668094,
				0.85557411, 0.0356366, 0.69422093,
			}),
			x: mat64.NewVector(2, []float64{0.6, 0.9}),
			t: mat64.NewVector(3, []float64{0, 0, 1}),
			dW: mat64.NewDense(2, 3, []float64{
				0.2192, 0.1435, -0.3628,
				0.3288, 0.2153, -0.5442,
			}),
		},
	} {
		impl := NewSimple2NN(c.W)
		nn := NewNeuralNet(impl)

		f := func(w *mat64.Dense) float64 {
			return nn.Loss(c.x, c.t)
		}
		dW := NumericalGradM(f, c.W)
		if !mat64.EqualApprox(c.dW, dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, dW)
		}

		nn.Loss(c.x, c.t)
		nn.BackProp()
		if !mat64.EqualApprox(impl.affine.DWeight, c.dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, impl.affine.DWeight)
		}

		g := func(w *mat64.Vector) float64 {
			return nn.Loss(c.x, c.t)
		}
		dB := NumericalGrad(g, impl.affine.Bias)
		if !mat64.EqualApprox(impl.affine.DBias, dB, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, impl.affine.DBias, dB)
		}
	}
}

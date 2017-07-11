package gocnn

/*
import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

type BatchNNSimple struct {
	affine *BatchAffineLayer
	relu   *BatchReLULayer
	last   *BatchSoftMaxWithLoss
}

func NewBatchNNSimple(w *mat64.Dense) *BatchNNSimple {
	_, c := w.Dims()
	b := mat64.NewVector(c, nil)
	al := NewBatchAffineLayer(w, b, NewMomentum(0.1, 0.1))
	return &BatchNNSimple{
		affine: al,
		relu:   &BatchReLULayer{},
		last:   &BatchSoftMaxWithLoss{},
	}
}

var _ BatchNeuralNetLayers = &BatchNNSimple{}

func (nn *BatchNNSimple) Layers() []BatchLayer {
	return []BatchLayer{nn.affine, nn.relu}
}
func (nn *BatchNNSimple) Last() BatchLastLayer {
	return nn.last
}

func TestBatchNNSimple(t *testing.T) {
	for _, c := range []struct {
		title string
		W     *mat64.Dense
		x     mat64.Matrix
		t     mat64.Matrix
		dW    mat64.Matrix
	}{
		{
			title: "TestNNGrad",
			W: mat64.NewDense(2, 3, []float64{
				0.47355232, 0.9977393, 0.84668094,
				0.85557411, 0.0356366, 0.69422093,
			}),
			x: mat64.NewDense(1, 2, []float64{0.6, 0.9}),
			t: mat64.NewDense(1, 3, []float64{0, 0, 1}),
			dW: mat64.NewDense(2, 3, []float64{
				0.2192, 0.1435, -0.3628,
				0.3288, 0.2153, -0.5442,
			}),
		},
	} {
		layers := NewBatchNNSimple(c.W)
		nn := &BatchNeuralNet{layers}

		f := func(w *mat64.Dense) float64 {
			return nn.Loss(c.x, c.t)
		}
		dW := NumericalGradM(f, c.W)
		if !mat64.EqualApprox(c.dW, dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, dW)
		}

		nn.Loss(c.x, c.t)
		nn.BackProp()
		if !mat64.EqualApprox(layers.affine.DWeight, c.dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, layers.affine.DWeight)
		}

		g := func(w *mat64.Vector) float64 {
			return nn.Loss(c.x, c.t)
		}
		dB := NumericalGrad(g, layers.affine.Bias)
		if !mat64.EqualApprox(layers.affine.DBias, dB, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, layers.affine.DBias, dB)
		}
	}
}
*/

package gocnn

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math/rand"
	"testing"

	"github.com/ajiyoshi/gocnn/mnist"
)

func TestBackPropGrad(t *testing.T) {
	m, err := mnist.NewMnist("./train-images-idx3-ubyte.idx", "./train-labels-idx1-ubyte.idx")
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	img := m.Images
	len := img.Rows * img.Cols
	impl := NewTwoLayerNN(len, 50, 10, NewMomentumFactory(0.1, 0.1))
	nn := NewNeuralNet(impl)

	buf := make([]float64, len)

	data, label := m.At(rand.Intn(m.Images.Num))
	mnist.LoadVec(data, buf)
	x := mat64.NewVector(len, buf)
	l := mnist.LoadLabel(label)

	nn.Loss(x, l)
	nn.BackProp()

	f := func(w *mat64.Dense) float64 {
		return nn.Loss(x, l)
	}
	dW := NumericalGradM(f, impl.affine1.Weight)

	sub := mat64.DenseCopyOf(impl.affine1.DWeight)
	sub.Sub(sub, dW)
	max := mat64.Max(sub)
	if max > 0.01 {
		t.Fatal(dW, impl.affine1.DWeight)
	}
	fmt.Printf("max diff %f\n", max)
}

func TestArgmax(t *testing.T) {
	for _, c := range []struct {
		title  string
		x      *mat64.Vector
		expect int
	}{
		{
			title:  "TestArgmax(",
			x:      mat64.NewVector(3, []float64{0, 0, 1}),
			expect: 2,
		},
	} {
		actual := ArgmaxV(c.x)
		if actual != c.expect {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.expect, actual)
		}
	}
}

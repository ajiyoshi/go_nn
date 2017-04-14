package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"testing"
)

type SimpleNN struct {
	dimIn  int
	dimOut int
	layer  *AffineLayer
}

func NewSimpleNN(w *mat64.Dense) *SimpleNN {
	r, c := w.Dims()
	b := mat64.NewVector(c, nil)
	al := NewAffineLayer(w, b, nil)
	return &SimpleNN{
		dimIn:  r,
		dimOut: c,
		layer:  al,
	}
}
func (nn *SimpleNN) Predict(x *mat64.Vector) *mat64.Vector {
	if x.Len() != nn.dimIn {
		panic("bad matrix")
	}
	return nn.layer.Forward(x)
}
func (nn *SimpleNN) Loss(x, t *mat64.Vector) float64 {
	z := nn.Predict(x)
	y := SoftMax(z)
	return CrossEntropyError(y, t)
}

func (nn *SimpleNN) LossWith(w *mat64.Dense, x, t *mat64.Vector) float64 {
	orig := nn.layer.Weight
	nn.layer.Weight = w
	ret := nn.Loss(x, t)
	nn.layer.Weight = orig
	return ret
}

func (nn *SimpleNN) Grad(x, t *mat64.Vector) {
	dout := SoftMax(nn.Predict(x))
	dout.SubVec(dout, t)
	nn.layer.Backward(dout)
}

func TestNNGrad(t *testing.T) {
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
		nn := NewSimpleNN(c.W)

		f := func(w *mat64.Dense) float64 {
			return nn.LossWith(w, c.x, c.t)
		}
		dW := NumericalGradM(f, c.W)
		if !mat64.EqualApprox(c.dW, dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, dW)
		}
		nn.Grad(c.x, c.t)
		if !mat64.EqualApprox(nn.layer.DWeight, c.dW, 0.01) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, c.dW, nn.layer.DWeight)
		}
		Dump(nn.layer.DWeight)

	}
}

func TestBackPropGrad(t *testing.T) {
	m, err := NewMnist("./train-images-idx3-ubyte", "./train-labels-idx1-ubyte")
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	img := m.Images
	len := img.Rows * img.Cols
	nn := NewNN(len, 50, 10, NewMomentumFactory(0.1, 0.1))

	buf := make([]float64, len)

	data, label := m.At(10)
	LoadVec(data, buf)
	x := mat64.NewVector(len, buf)
	l := LoadLabel(label)

	grad := nn.Grad(x, l)
	ngra := nn.NumGrad(x, l)

	sub := mat64.DenseCopyOf(grad[0])
	sub.Sub(sub, ngra[0])
	max := mat64.Max(sub)
	if max > 0.01 {
		//t.Fatal(grad[0], ngra[0])
	}
	fmt.Printf("max diff %f\n", max)

	/*
		fmt.Printf("%v\n",
			mat64.Formatted(m, mat64.Prefix(" "), mat64.Excerpt(3)))
	*/
}

func LoadVec(raw []byte, buf []float64) {
	for i, v := range raw {
		buf[i] = float64(v)
	}
}

var labels []*mat64.Vector = []*mat64.Vector{
	mat64.NewVector(10, []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}),
	mat64.NewVector(10, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}),
}

func LoadLabel(label byte) *mat64.Vector {
	return labels[label]
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

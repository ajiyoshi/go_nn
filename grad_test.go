package main

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func VF2D(f func(a, b float64) float64) VectorFunction {
	return func(x *mat64.Vector) float64 {
		if x.Len() != 2 {
			panic("bad matrix")
		}
		x0 := x.At(0, 0)
		x1 := x.At(1, 0)
		return f(x0, x1)
	}
}

func TestGrad(t *testing.T) {
	f := VF2D(func(x0, x1 float64) float64 {
		return x0*x0 + x1*x1
	})
	for _, c := range []struct {
		title string
		pt    []float64
		grad  []float64
	}{
		{
			"(3, 4)",
			[]float64{3, 4},
			[]float64{6, 8},
		},
		{
			"(0, 2)",
			[]float64{0, 2},
			[]float64{0, 4},
		},
		{
			"(3, 0)",
			[]float64{3, 0},
			[]float64{6, 0},
		},
	} {
		pt := mat64.NewVector(2, c.pt)
		expect := mat64.NewVector(2, c.grad)
		actual := NumericalGrad(f, pt)
		if !mat64.EqualApprox(actual, expect, 0.001) {
			t.Fatalf("%s expect(%v) but got (%v)", c.title, actual, expect)
		}
	}

}

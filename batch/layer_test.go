package gocnn

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestAffineLayer(t *testing.T) {
	cases := []struct {
		title  string
		W      *mat64.Dense
		B      *mat64.Vector
		x      mat64.Matrix
		y      mat64.Matrix
		dout   mat64.Matrix
		expect mat64.Matrix
		dW     mat64.Matrix
		dB     mat64.Matrix
	}{
		{
			title: "TestAffineLayer",
			W: mat64.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
			B: mat64.NewVector(2, []float64{9, 12}),
			x: mat64.NewDense(1, 3, []float64{
				1, 2, 3,
			}),
			// y = x * W + B
			y: mat64.NewDense(1, 2, []float64{
				9 + 1*1 + 2*3 + 3*5, 12 + 1*2 + 2*4 + 3*6,
			}),
			dout: mat64.NewDense(1, 2, []float64{
				1, 2,
			}),
			// expect = dout * W.T()
			expect: mat64.NewDense(1, 3, []float64{
				1*1 + 2*2, 1*3 + 2*4, 1*5 + 2*6,
			}),
			// dW = x.T() * dout
			dW: mat64.NewDense(3, 2, []float64{
				1 * 1, 1 * 2,
				2 * 1, 2 * 2,
				3 * 1, 3 * 2,
			}),
			// dB = np.sum(dout, axis=0)
			dB: mat64.NewVector(2, []float64{
				1, 2,
			}),
		},
		{
			title: "TestAffineLayer2",
			W: mat64.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
			B: mat64.NewVector(2, []float64{9, 12}),
			x: mat64.NewDense(3, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),
			// y = x * W + B
			y: mat64.NewDense(3, 2, []float64{
				9 + 1*1 + 2*3 + 3*5, 12 + 1*2 + 2*4 + 3*6,
				9 + 4*1 + 5*3 + 6*5, 12 + 4*2 + 5*4 + 6*6,
				9 + 7*1 + 8*3 + 9*5, 12 + 7*2 + 8*4 + 9*6,
			}),
			dout: mat64.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				4, 5,
			}),
			// expect = dout * W.T()
			expect: mat64.NewDense(3, 3, []float64{
				1*1 + 2*2, 1*3 + 2*4, 1*5 + 2*6,
				3*1 + 4*2, 3*3 + 4*4, 3*5 + 4*6,
				4*1 + 5*2, 4*3 + 5*4, 4*5 + 5*6,
			}),
			// x.T() * dout
			dW: mat64.NewDense(3, 2, []float64{
				1*1 + 4*3 + 7*4, 1*2 + 4*4 + 7*5,
				2*1 + 5*3 + 8*4, 2*2 + 5*4 + 8*5,
				3*1 + 6*3 + 9*4, 3*2 + 6*4 + 9*5,
			}),
			// np.sum(dout, axis=0)
			dB: mat64.NewVector(2, []float64{
				1 + 3 + 4, 2 + 4 + 5,
			}),
		},
	}

	for _, c := range cases {
		l := NewAffineLayer(c.W, c.B, nil)
		y := l.Forward(c.x)
		if !mat64.EqualApprox(y, c.y, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.y, y)
		}
		actual := l.Backward(c.dout)
		if !mat64.EqualApprox(actual, c.expect, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.expect, actual)
		}
		if !mat64.EqualApprox(l.DWeight, c.dW, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.dW, l.DWeight)
		}
		if !mat64.EqualApprox(l.DBias, c.dB, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.dB, l.DBias)
		}
	}
}

func TestReLULayer(t *testing.T) {
	cases := []struct {
		title  string
		x      mat64.Matrix
		y      mat64.Matrix
		dout   mat64.Matrix
		expect mat64.Matrix
	}{
		{
			title: "TestReLULayer",
			x: mat64.NewDense(2, 3, []float64{
				1, -1, 0,
				-1, 0, -1,
			}),
			y: mat64.NewDense(2, 3, []float64{
				1, 0, 0,
				0, 0, 0,
			}),
			dout: mat64.NewDense(2, 3, []float64{
				2, 2, 2,
				2, 2, 2,
			}),
			expect: mat64.NewDense(2, 3, []float64{
				2, 0, 2,
				0, 2, 0,
			}),
		},
	}

	for _, c := range cases {
		l := &ReLULayer{}
		y := l.Forward(c.x)
		if !mat64.EqualApprox(y, c.y, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.y, y)
		}
		actual := l.Backward(c.dout)
		if !mat64.EqualApprox(c.expect, actual, 0.01) {
			t.Fatalf("%s expect %v but got %v", c.title, c.expect, actual)
		}
	}
}

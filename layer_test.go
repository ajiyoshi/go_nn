package gocnn

import (
	"github.com/ajiyoshi/gocnn/nd"
	mat "github.com/gonum/matrix/mat64"
	"testing"
)

func zeros(n int) *mat.Vector {
	return mat.NewVector(n, make([]float64, n))
}

func mustLoad(path string) Image {
	return NewArrayImage(nd.Must(nd.Load(path)))
}

func TestTest(t *testing.T) {
	cases := []struct {
		msg      string
		generate func() (c *Convolution, x, expect Image)
	}{
		{
			msg: "",
			generate: func() (*Convolution, Image, Image) {
				x := mustLoad("t/x.mp")
				expect := mustLoad("t/x1.mp")
				w := mustLoad("t/W0.mp")

				conv := &Convolution{
					Weight: w,
					Bias:   mat.NewVector(w.Shape().N, nil),
					Stride: 1,
					Pad:    0,
				}

				return conv, x, expect
			},
		},
	}
	for _, c := range cases {
		layer, x, expect := c.generate()
		y := layer.Forward(x)
		if !y.Equal(expect) {
			t.Fatalf("expect \n%v got \n%v", expect, y)
		}
	}
}

func TestForward(t *testing.T) {

	cases := []struct {
		msg      string
		generate func() (c *Convolution, x, expect Image)
	}{
		{
			msg: "実際に実行してみた結果",
			generate: func() (*Convolution, Image, Image) {
				filterNum := 1
				filterSize := 3
				dataNum, chNum, xRow, xCol := 1, 1, 2, 2
				x := NewImages(&Shape{N: dataNum, Ch: chNum, Row: xRow, Col: xCol}, []float64{
					1, 2,
					3, 4,
				})

				w := NewImages(&Shape{N: filterNum, Ch: chNum, Row: filterSize, Col: filterSize}, []float64{
					0.01061144, 0.00930966, 0.00157138,
					0.01366734, 0.00596517, -0.00052856,
					-0.0022351, -0.00402149, -0.00019544,
				})
				bias := zeros(filterNum)
				conv := &Convolution{
					Weight: w,
					Bias:   bias,
					Stride: 1,
					Pad:    1,
				}
				expect := NewImages(&Shape{dataNum, filterNum, 2, 2}, []float64{
					-0.00793818, 0.00280642,
					0.02823369, 0.09409346,
				})
				return conv, x, expect
			},
		},
		{
			msg: "実際に実行してみた結果",
			generate: func() (*Convolution, Image, Image) {
				filterNum := 2
				filterSize := 3
				dataNum, chNum, xRow, xCol := 1, 1, 2, 2
				x := NewImages(&Shape{N: dataNum, Ch: chNum, Row: xRow, Col: xCol}, []float64{
					1, 2,
					3, 4,
				})

				w := NewImages(&Shape{N: filterNum, Ch: chNum, Row: filterSize, Col: filterSize}, []float64{
					-0.00205192, -0.01427015, 0.01118195,
					-0.00115402, 0.00920227, -0.00072591,
					0.00013398, 0.00050144, -0.01106872,

					-0.00434496, -0.00634229, 0.01542008,
					-0.01091955, 0.00087254, 0.00059587,
					-0.00545791, 0.00746039, 0.00338262,
				})
				bias := zeros(filterNum)
				conv := &Convolution{
					Weight: w,
					Bias:   bias,
					Stride: 1,
					Pad:    1,
				}
				expect := NewImages(&Shape{N: dataNum, Ch: filterNum, Row: 2, Col: 2}, []float64{
					-0.03502013, 0.01965824,
					0.0327969, 0.00275479,

					0.03797594, 0.00429336,
					0.02949898, -0.04629804,
				})
				return conv, x, expect
			},
		},
	}

	for _, c := range cases {
		layer, x, expect := c.generate()
		y := layer.Forward(x)
		if !y.Equal(expect) {
			t.Fatalf("expect \n%v got \n%v", expect, y)
		}
	}
}

func TestBackword(t *testing.T) {
	filterNum := 2
	filterSize := 3
	dataNum, chNum, xRow, xCol := 1, 1, 2, 2
	x := NewImages(&Shape{N: dataNum, Ch: chNum, Row: xRow, Col: xCol}, []float64{
		1, 2,
		3, 4,
	})

	w := NewImages(&Shape{N: filterNum, Ch: chNum, Row: filterSize, Col: filterSize}, []float64{
		-0.00431678, 0.00845254, 0.00188367,
		0.0036537, -0.00328172, 0.0070477,
		0.0162438, -0.00225665, -0.00838634,

		0.01078964, 0.01474864, 0.00210126,
		-0.01044749, -0.00207768, -0.00652951,
		-0.00074343, -0.00369562, -0.00204102,
	})
	bias := zeros(filterNum)
	conv := &Convolution{
		Weight: w,
		Bias:   bias,
		Stride: 1,
		Pad:    1,
	}
	y := conv.Forward(x)
	actual := conv.Backword(y)

	expect := NewArrayImage(nd.NewArray(nd.NewShape(1, 1, 2, 2), []float64{
		0.0006555762625782, 8.841391855100008e-05,
		0.0007738472404583999, 0.0006187933526248,
	}))
	if !expect.Equal(actual) {
		t.Fatalf("expect \n%v got \n%v", expect, actual)
	}
}

func TestPooling(t *testing.T) {
	cases := []struct {
		msg      string
		generate func() (p *Pooling, x, forward, backward Image)
	}{
		{
			msg: "",
			generate: func() (p *Pooling, x, forward, backward Image) {
				p = &Pooling{Row: 2, Col: 2, Stride: 1, Pad: 0}
				x = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 3, 3}, []float64{
					1, 2, 3,
					6, 5, 4,
					8, 9, 7,
				}))
				forward = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 2, 2}, []float64{
					6, 5,
					9, 9,
				}))
				backward = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 3, 3}, []float64{
					0, 0, 0,
					6, 5, 0,
					0, 18, 0,
				}))
				return p, x, forward, backward
			},
		},
		{
			msg: "",
			generate: func() (p *Pooling, x, forward, backward Image) {
				p = &Pooling{Row: 3, Col: 3, Stride: 1, Pad: 0}
				x = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 3, 3}, []float64{
					1, 2, 3,
					6, 5, 4,
					8, 9, 7,
				}))
				forward = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 1, 1}, []float64{
					9,
				}))
				backward = NewArrayImage(nd.NewArray(nd.Shape{1, 1, 3, 3}, []float64{
					0, 0, 0,
					0, 0, 0,
					0, 9, 0,
				}))
				return p, x, forward, backward
			},
		},
	}

	for _, c := range cases {
		pool, x, forward, backward := c.generate()
		y := pool.Forward(x)
		if !y.Equal(forward) {
			t.Fatalf("expect \n%v got \n%v", forward, y)
		}

		z := pool.Backword(y)
		if !z.Equal(backward) {
			t.Fatalf("expect \n%v got \n%v", backward, z)
		}
	}
}

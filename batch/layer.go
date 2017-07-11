package batch

import (
	"fmt"
	"github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn"
)

type Layer interface {
	Forward(mat64.Matrix) mat64.Matrix
	Backward(mat64.Matrix) mat64.Matrix
	Update()
}

type LastLayer interface {
	Forward(x, t mat64.Matrix) float64
	Backward(float64) mat64.Matrix
}

var (
	_ Layer     = &AffineLayer{}
	_ Layer     = &ReLULayer{}
	_ LastLayer = &SoftMaxWithLoss{}
)

type AffineLayer struct {
	dimIn     int
	dimOut    int
	Weight    *mat64.Dense
	Bias      *mat64.Vector
	DWeight   *mat64.Dense
	DBias     *mat64.Vector
	x         mat64.Matrix
	optimizer gocnn.Optimizer
}

func NewAffineLayer(w *mat64.Dense, b *mat64.Vector, o gocnn.Optimizer) *AffineLayer {
	r, c := w.Dims()
	if c != b.Len() {
		panic(fmt.Sprintf("cols:%d, b.Len():%d", c, b.Len()))
	}
	return &AffineLayer{
		dimIn:     r,
		dimOut:    c,
		Weight:    w,
		Bias:      b,
		DWeight:   mat64.NewDense(r, c, nil),
		DBias:     mat64.NewVector(b.Len(), nil),
		optimizer: o,
	}
}

func NewAffine(weight float64, input, output int, op gocnn.Optimizer) *AffineLayer {
	w := gocnn.RandamDense(input, output)
	w.Scale(weight, w)
	b := mat64.NewVector(output, nil)
	return NewAffineLayer(w, b, op)
}

// x : (N, dimIN)
// W : (dimIN, dimOut)
// ret : (N, dimOut)
// b : dimOut
func (l *AffineLayer) Forward(x mat64.Matrix) mat64.Matrix {
	_, c := x.Dims()
	if c != l.dimIn {
		panic(fmt.Sprintf("expect %d but got %d", l.dimIn, c))
	}
	l.x = x
	var ret mat64.Dense
	ret.Mul(x, l.Weight)
	ret.Apply(func(i, j int, val float64) float64 {
		return l.Bias.At(j, 0) + val
	}, &ret)
	return &ret
}

// dout : (N, dimOut)
// weight.T : (dimOut, dimIn)
// ret : (N, dimIn)
// b : dimOut
func (l *AffineLayer) Backward(dout mat64.Matrix) mat64.Matrix {
	r, c := dout.Dims()
	N, _ := l.x.Dims()
	if r != N || c != l.dimOut {
		panic(fmt.Sprintf("expect (%d, %d) but got (%d, %d)", N, l.dimOut, r, c))
	}
	var dx mat64.Dense
	dx.Mul(dout, l.Weight.T())

	l.DWeight.Mul(l.x.T(), dout)
	gocnn.SumCols(dout, l.DBias)

	return &dx
}

func (l *AffineLayer) Update() {
	l.optimizer.UpdateWeight(l.Weight, l.DWeight)
	l.optimizer.UpdateBias(l.Bias, l.DBias)
}

type ReLULayer struct {
	mask *mat64.Dense
}

func NewReLU() *ReLULayer {
	return &ReLULayer{}
}

func (l *ReLULayer) Forward(x mat64.Matrix) mat64.Matrix {
	r, c := x.Dims()
	l.mask = mat64.NewDense(r, c, nil)
	l.mask.Apply(func(i, j int, v float64) float64 {
		if v < 0 {
			return 0
		} else {
			return 1
		}
	}, x)
	var ret mat64.Dense
	ret.MulElem(x, l.mask)
	return &ret
}
func (l *ReLULayer) Backward(dout mat64.Matrix) mat64.Matrix {
	l.mask.MulElem(l.mask, dout)
	return l.mask
}
func (l *ReLULayer) Update() {
}

type SoftMaxWithLoss struct {
	loss float64
	y    mat64.Matrix
	t    mat64.Matrix
}

func NewSoftMaxWithLoss() *SoftMaxWithLoss {
	return &SoftMaxWithLoss{}
}

func (l *SoftMaxWithLoss) Forward(x, t mat64.Matrix) float64 {
	l.t = t
	l.y = gocnn.SoftMax(x)
	l.loss = gocnn.CrossEntropyError(l.y, t)
	return l.loss
}

func (l *SoftMaxWithLoss) Backward(dout float64) mat64.Matrix {
	r, _ := l.y.Dims()
	var dx mat64.Dense
	dx.Sub(l.y, l.t)
	dx.Scale(1.0/float64(r), &dx)
	return &dx
}

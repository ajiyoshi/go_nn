package batch

import (
	"fmt"
	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/optimizer"
)

type Layer interface {
	Forward(mat.Matrix) mat.Matrix
	Backward(mat.Matrix) mat.Matrix
	Update()
}

type LastLayer interface {
	Forward(x, t mat.Matrix) float64
	Backward(float64) mat.Matrix
}

var (
	_ Layer     = &AffineLayer{}
	_ Layer     = &ReLULayer{}
	_ LastLayer = &SoftMaxWithLoss{}
)

type AffineLayer struct {
	dimIn     int
	dimOut    int
	Weight    *mat.Dense
	Bias      *mat.Vector
	DWeight   *mat.Dense
	DBias     *mat.Vector
	x         mat.Matrix
	optimizer optimizer.Optimizer
}

func NewAffineLayer(w *mat.Dense, b *mat.Vector, o optimizer.Optimizer) *AffineLayer {
	r, c := w.Dims()
	if c != b.Len() {
		panic(fmt.Sprintf("cols:%d, b.Len():%d", c, b.Len()))
	}
	return &AffineLayer{
		dimIn:     r,
		dimOut:    c,
		Weight:    w,
		Bias:      b,
		DWeight:   mat.NewDense(r, c, nil),
		DBias:     mat.NewVector(b.Len(), nil),
		optimizer: o,
	}
}

func NewAffine(weight float64, input, output int, op optimizer.Optimizer) *AffineLayer {
	w := matrix.RandamDense(input, output)
	w.Scale(weight, w)
	b := mat.NewVector(output, nil)
	return NewAffineLayer(w, b, op)
}

// x : (N, dimIN)
// W : (dimIN, dimOut)
// ret : (N, dimOut)
// b : dimOut
func (l *AffineLayer) Forward(x mat.Matrix) mat.Matrix {
	_, c := x.Dims()
	if c != l.dimIn {
		panic(fmt.Sprintf("expect %d but got %d", l.dimIn, c))
	}
	l.x = x
	var ret mat.Dense
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
func (l *AffineLayer) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	N, _ := l.x.Dims()
	if r != N || c != l.dimOut {
		panic(fmt.Sprintf("expect (%d, %d) but got (%d, %d)", N, l.dimOut, r, c))
	}
	var dx mat.Dense
	dx.Mul(dout, l.Weight.T())

	l.DWeight.Mul(l.x.T(), dout)
	matrix.SumCols(dout, l.DBias)

	return &dx
}

func (l *AffineLayer) Update() {
	l.optimizer.UpdateWeight(l.Weight, l.DWeight)
	l.optimizer.UpdateBias(l.Bias, l.DBias)
}

type ReLULayer struct {
	mask *mat.Dense
}

func NewReLU() *ReLULayer {
	return &ReLULayer{}
}

func (l *ReLULayer) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	l.mask = mat.NewDense(r, c, nil)
	l.mask.Apply(func(i, j int, v float64) float64 {
		if v < 0 {
			return 0
		} else {
			return 1
		}
	}, x)
	var ret mat.Dense
	ret.MulElem(x, l.mask)
	return &ret
}
func (l *ReLULayer) Backward(dout mat.Matrix) mat.Matrix {
	l.mask.MulElem(l.mask, dout)
	return l.mask
}
func (l *ReLULayer) Update() {
}

type SoftMaxWithLoss struct {
	loss float64
	y    mat.Matrix
	t    mat.Matrix
}

func NewSoftMaxWithLoss() *SoftMaxWithLoss {
	return &SoftMaxWithLoss{}
}

func (l *SoftMaxWithLoss) Forward(x, t mat.Matrix) float64 {
	l.t = t
	l.y = matrix.SoftMax(x)
	l.loss = matrix.CrossEntropyError(l.y, t)
	return l.loss
}

func (l *SoftMaxWithLoss) Backward(dout float64) mat.Matrix {
	r, _ := l.y.Dims()
	var dx mat.Dense
	dx.Sub(l.y, l.t)
	dx.Scale(1.0/float64(r), &dx)
	return &dx
}

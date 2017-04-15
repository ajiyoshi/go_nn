package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

type BatchLayer interface {
	Forward(mat64.Matrix) mat64.Matrix
	Backward(mat64.Matrix) mat64.Matrix
	Update()
}

type BatchLastLayer interface {
	Forward(x, t mat64.Matrix) float64
	Backward(float64) mat64.Matrix
}

var (
	_ BatchLayer     = &BatchAffineLayer{}
	_ BatchLayer     = &BatchReLULayer{}
	_ BatchLastLayer = &BatchSoftMaxWithLoss{}
)

type BatchAffineLayer struct {
	dimIn     int
	dimOut    int
	Weight    *mat64.Dense
	Bias      *mat64.Vector
	DWeight   *mat64.Dense
	DBias     *mat64.Vector
	x         mat64.Matrix
	optimizer Optimizer
}

func NewBatchAffineLayer(w *mat64.Dense, b *mat64.Vector, o Optimizer) *BatchAffineLayer {
	r, c := w.Dims()
	if c != b.Len() {
		panic(fmt.Sprintf("cols:%d, b.Len():%d", c, b.Len()))
	}
	return &BatchAffineLayer{
		dimIn:     r,
		dimOut:    c,
		Weight:    w,
		Bias:      b,
		DWeight:   mat64.NewDense(r, c, nil),
		DBias:     mat64.NewVector(b.Len(), nil),
		optimizer: o,
	}
}

// x : (N, dimIN)
// W : (dimIN, dimOut)
// ret : (N, dimOut)
// b : dimOut
func (l *BatchAffineLayer) Forward(x mat64.Matrix) mat64.Matrix {
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
func (l *BatchAffineLayer) Backward(dout mat64.Matrix) mat64.Matrix {
	r, c := dout.Dims()
	N, _ := l.x.Dims()
	if r != N || c != l.dimOut {
		panic(fmt.Sprintf("expect (%d, %d) but got (%d, %d)", N, l.dimOut, r, c))
	}
	var dx mat64.Dense
	dx.Mul(dout, l.Weight.T())

	l.DWeight.Mul(l.x.T(), dout)
	SumCols(dout, l.DBias)

	return &dx
}

func (l *BatchAffineLayer) Update() {
	l.optimizer.UpdateWeight(l.Weight, l.DWeight)
	l.optimizer.UpdateBias(l.Bias, l.DBias)
}

type BatchReLULayer struct {
	mask *mat64.Dense
}

func (l *BatchReLULayer) Forward(x mat64.Matrix) mat64.Matrix {
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
func (l *BatchReLULayer) Backward(dout mat64.Matrix) mat64.Matrix {
	l.mask.MulElem(l.mask, dout)
	return l.mask
}
func (l *BatchReLULayer) Update() {
}

type BatchSoftMaxWithLoss struct {
	loss float64
	y    mat64.Matrix
	t    mat64.Matrix
}

func (l *BatchSoftMaxWithLoss) Forward(x, t mat64.Matrix) float64 {
	l.t = t
	l.y = SoftMax(x)
	l.loss = CrossEntropyError(l.y, t)
	return l.loss
}

func (l *BatchSoftMaxWithLoss) Backward(dout float64) mat64.Matrix {
	var dx mat64.Dense
	dx.Sub(l.y, l.t)
	return &dx
}

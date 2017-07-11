package gocnn

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

type Layer interface {
	Forward(*mat64.Vector) *mat64.Vector
	Backward(*mat64.Vector) *mat64.Vector
	Update()
}

type LastLayer interface {
	Forward(x, t *mat64.Vector) float64
	Backward(float64) *mat64.Vector
}

var (
	_ Layer = &AffineLayer{}
	_ Layer = &ReLULayer{}
)

type AffineLayer struct {
	dimIn     int
	dimOut    int
	Weight    *mat64.Dense
	Bias      *mat64.Vector
	DWeight   *mat64.Dense
	DBias     *mat64.Vector
	x         *mat64.Vector
	optimizer Optimizer
}

func NewAffineLayer(w *mat64.Dense, b *mat64.Vector, o Optimizer) *AffineLayer {
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
		x:         mat64.NewVector(r, nil),
		optimizer: o,
	}
}

func (l *AffineLayer) Forward(x *mat64.Vector) *mat64.Vector {
	if x.Len() != l.dimIn {
		panic(fmt.Sprintf("expect %d but got %d", l.dimIn, x.Len()))
	}
	l.x.CloneVec(x)
	ret := mat64.NewVector(l.dimOut, nil)
	ret.MulVec(l.Weight.T(), x)
	ret.AddVec(ret, l.Bias)
	return ret
}

func (l *AffineLayer) Backward(dout *mat64.Vector) *mat64.Vector {
	if dout.Len() != l.dimOut {
		panic(fmt.Sprintf("expect %d but got %d", l.dimOut, dout.Len()))
	}
	dx := mat64.NewVector(l.dimIn, nil)
	dx.MulVec(l.Weight, dout)

	l.DWeight.Mul(l.x, dout.T())
	l.DBias.CloneVec(dout)

	return dx
}

func (l *AffineLayer) Update() {
	l.optimizer.UpdateWeight(l.Weight, l.DWeight)
	l.optimizer.UpdateBias(l.Bias, l.DBias)
}

type ReLULayer struct {
	mask *mat64.Vector
}

func (l *ReLULayer) Forward(x *mat64.Vector) *mat64.Vector {
	l.mask = VecClone(x)
	VecApply(l.mask, func(val float64) float64 {
		if val < 0 {
			return 0
		} else {
			return 1
		}
	})
	ret := mat64.NewVector(x.Len(), nil)
	ret.MulElemVec(x, l.mask)
	return ret
}
func (l *ReLULayer) Backward(dout *mat64.Vector) *mat64.Vector {
	dout.MulElemVec(dout, l.mask)
	return dout
}
func (l *ReLULayer) Update() {
}

type SoftMaxWithLoss struct {
	loss float64
	y    *mat64.Vector
	t    *mat64.Vector
}

func (l *SoftMaxWithLoss) Forward(x, t *mat64.Vector) float64 {
	l.t = VecClone(t)
	l.y = SoftMaxV(x)
	l.loss = CrossEntropyError(l.y, t)
	return l.loss
}

func (l *SoftMaxWithLoss) Backward(dout float64) *mat64.Vector {
	dx := mat64.NewVector(l.y.Len(), nil)
	dx.SubVec(l.y, l.t)
	return dx
}

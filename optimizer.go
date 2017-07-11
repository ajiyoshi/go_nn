package gocnn

import (
	"github.com/gonum/matrix/mat64"
)

type Optimizer interface {
	UpdateWeight(param, grad *mat64.Dense)
	UpdateBias(param, grad *mat64.Vector)
}

var _ Optimizer = &Momentum{}

type Momentum struct {
	Lr       float64
	Momentum float64
	vW       *mat64.Dense
	vB       *mat64.Vector
}

type OptimizerFactory func() Optimizer

func NewMomentumFactory(lr, mo float64) OptimizerFactory {
	return func() Optimizer {
		return NewMomentum(lr, mo)
	}
}

func NewMomentum(lr, mo float64) *Momentum {
	return &Momentum{
		Lr:       lr,
		Momentum: mo,
	}
}

func (o *Momentum) UpdateWeight(param, grad *mat64.Dense) {
	if o.vW == nil {
		r, c := param.Dims()
		o.vW = mat64.NewDense(r, c, nil)
	}
	v := o.vW
	update := func(i, j int, x float64) float64 {
		return o.Momentum*x - o.Lr*grad.At(i, j)
	}
	v.Apply(update, v)
	param.Add(param, v)
}

func (o *Momentum) UpdateBias(param, grad *mat64.Vector) {
	if o.vB == nil {
		len := param.Len()
		o.vB = mat64.NewVector(len, nil)
	}
	v := o.vB
	v.ScaleVec(o.Momentum, v)
	v.AddScaledVec(v, -o.Lr, grad)
	param.AddVec(param, v)
}

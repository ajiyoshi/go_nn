package optimizer

import (
	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/nd"
)

type Optimizer interface {
	UpdateWeight(param, grad *mat.Dense)
	UpdateWeightArray(param, grad nd.Array)
	UpdateBias(param, grad *mat.Vector)
}

var _ Optimizer = &Momentum{}

type Momentum struct {
	Lr       float64
	Momentum float64
	vW       *mat.Dense
	vB       *mat.Vector
	vWa      nd.Array
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

func (o *Momentum) UpdateWeight(param, grad *mat.Dense) {
	if o.vW == nil {
		r, c := param.Dims()
		o.vW = mat.NewDense(r, c, nil)
	}
	v := o.vW
	update := func(i, j int, x float64) float64 {
		return o.Momentum*x - o.Lr*grad.At(i, j)
	}
	v.Apply(update, v)
	param.Add(param, v)
}

func (o *Momentum) UpdateWeightArray(param, grad nd.Array) {
	if !param.Shape().Equals(grad.Shape()) {
		panic("shape should be same")
	}
	if o.vWa == nil {
		o.vWa = nd.Zeros(param.Shape())
	}
	v := o.vWa
	v.Scale(o.Momentum).AddEach(grad.Scale(-o.Lr))
	param.AddEach(v)
}

func (o *Momentum) UpdateBias(param, grad *mat.Vector) {
	if o.vB == nil {
		len := param.Len()
		o.vB = mat.NewVector(len, nil)
	}
	v := o.vB
	v.ScaleVec(o.Momentum, v)
	v.AddScaledVec(v, -o.Lr, grad)
	param.AddVec(param, v)
}

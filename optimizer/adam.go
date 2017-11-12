package optimizer

import (
	mat "github.com/gonum/matrix/mat64"
	"math"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/nd"
)

type ArrayAdam struct {
	lr    float64
	beta1 float64
	beta2 float64
	iter  float64
	m     nd.Array
	v     nd.Array
}
type VectorAdam struct {
	lr    float64
	beta1 float64
	beta2 float64
	iter  float64
	m     *mat.Vector
	v     *mat.Vector
}
type MatrixAdam struct {
	lr    float64
	beta1 float64
	beta2 float64
	iter  float64
	m     *mat.Dense
	v     *mat.Dense
}

type Adam struct {
	array  ArrayAdam
	vector VectorAdam
	matrix MatrixAdam
}

var (
	_ Optimizer = (*Adam)(nil)
)

func NewAdam(lr, beta1, beta2 float64) OptimizerFactory {
	return func() Optimizer {
		return &Adam{
			array:  ArrayAdam{lr: lr, beta1: beta1, beta2: beta2},
			vector: VectorAdam{lr: lr, beta1: beta1, beta2: beta2},
			matrix: MatrixAdam{lr: lr, beta1: beta1, beta2: beta2},
		}
	}
}

func (o *Adam) UpdateWeightArray(param, grad nd.Array) {
	o.array.Update(param, grad)
}
func (o *Adam) UpdateWeight(param, grad *mat.Dense) {
	o.matrix.Update(param, grad)
}
func (o *Adam) UpdateBias(param, grad *mat.Vector) {
	o.vector.Update(param, grad)
}

func calcScale(iter, beta1, beta2 float64) float64 {
	return math.Sqrt(1.0-math.Pow(beta2, iter)) / (1.0 - math.Pow(beta1, iter))
}

func (o *ArrayAdam) Update(param, grad nd.Array) {
	if o.m == nil {
		o.m = nd.Zeros(param.Shape())
		o.v = nd.Zeros(param.Shape())
	}

	o.iter++
	lr := o.lr * calcScale(o.iter, o.beta1, o.beta2)

	o.m.AddEach(grad.Clone().SubEach(o.m).Scale(1 - o.beta1))
	o.v.AddEach(grad.Map(square).SubEach(o.v).Scale(1 - o.beta2))

	param.SubEach(o.m.Clone().Scale(lr).DivEach(o.v.Map(sqrt)))
}

func (o *VectorAdam) Update(param, grad *mat.Vector) {
	length := param.Len()
	if o.m == nil {
		o.m = mat.NewVector(length, nil)
		o.v = mat.NewVector(length, nil)
	}
	o.iter++
	lr := o.lr * calcScale(o.iter, o.beta1, o.beta2)

	a := mat.NewVector(length, nil)
	a.SubVec(grad, o.m)
	o.m.AddScaledVec(o.m, 1-o.beta1, a)

	a.MulElemVec(grad, grad)
	a.SubVec(a, o.v)
	o.v.AddScaledVec(o.v, 1-o.beta2, a)

	b := mat.NewVector(length, nil)
	b.CloneVec(o.v)
	matrix.VecApply(b, sqrt)

	a.DivElemVec(o.m, b)
	param.AddScaledVec(param, -lr, a)
}

func (o *MatrixAdam) Update(param, grad *mat.Dense) {
	r, c := param.Dims()
	if o.m == nil {
		o.m = mat.NewDense(r, c, nil)
		o.v = mat.NewDense(r, c, nil)
	}

	o.iter++
	lr := o.lr * calcScale(o.iter, o.beta1, o.beta2)

	a := mat.NewDense(r, c, nil)

	a.Sub(grad, o.m)
	a.Scale(1-o.beta1, a)
	o.m.Add(o.m, a)

	a.MulElem(grad, grad)
	a.Sub(a, o.v)
	a.Scale(1-o.beta2, a)
	o.v.Add(o.v, a)

	b := mat.NewDense(r, c, nil)
	b.Apply(func(i, j int, x float64) float64 {
		return sqrt(x)
	}, o.v)

	a.DivElem(o.m, b)
	a.Scale(-lr, a)
	param.Add(param, a)
}

func square(a float64) float64 {
	return a * a
}
func sqrt(a float64) float64 {
	if a < 0 {
		panic("hoge")
	}
	return math.Sqrt(a) + 1e-7
}

package matrix

import (
	"github.com/gonum/matrix/mat64"
	"math"
)

type VectorFunction func(*mat64.Vector) float64
type MatrixFunction func(*mat64.Dense) float64

func dx(x float64) float64 {
	if x == 0.0 {
		return 0.001
	} else {
		return math.Abs(x * 0.001)
	}
}

func NumericalGrad(f VectorFunction, x *mat64.Vector) *mat64.Vector {
	grad := mat64.NewVector(x.Len(), nil)

	for i := 0; i < x.Len(); i++ {
		tmp := x.At(i, 0)
		h := dx(tmp)

		x.SetVec(i, tmp+h)
		fxh1 := f(x)

		x.SetVec(i, tmp-h)
		fxh2 := f(x)

		grad.SetVec(i, (fxh1-fxh2)/(h*2))
	}

	return grad
}

func NumericalGradM(f MatrixFunction, x *mat64.Dense) mat64.Matrix {
	r, c := x.Dims()
	grad := mat64.NewDense(r, c, nil)

	grad.Apply(func(i, j int, tmp float64) float64 {
		h := dx(tmp)

		x.Set(i, j, tmp+h)
		fxh1 := f(x)

		x.Set(i, j, tmp-h)
		fxh2 := f(x)

		ret := (fxh1 - fxh2) / (h * 2)

		x.Set(i, j, tmp)

		return ret

	}, x)

	return grad
}

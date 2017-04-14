package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

func CrossEntropyError(y, t *mat64.Vector) float64 {
	const delta = 1e-7
	tmp := VecClone(y)
	VecApply(tmp, func(x float64) float64 {
		return math.Log(x + delta)
	})
	return -mat64.Dot(tmp, t)
}

func SoftMax(v *mat64.Vector) *mat64.Vector {
	ret := VecClone(v)

	max := mat64.Max(ret)
	VecAddEach(ret, -max)
	VecApply(ret, math.Exp)
	denom := 1.0 / mat64.Sum(ret)
	ret.ScaleVec(denom, ret)
	return ret
}

func ArgmaxV(v *mat64.Vector) int {
	len := v.Len()
	switch len {
	case 0:
		panic("bad matrix")
	case 1:
		return 0
	}

	max := v.At(0, 0)
	ret := 0

	for i := 1; i < v.Len(); i++ {
		if v.At(i, 0) > max {
			max = v.At(i, 0)
			ret = i
		}
	}
	return ret
}

func VecApply(v *mat64.Vector, f func(float64) float64) {
	raw := v.RawVector()
	for i, x := range raw.Data {
		v.SetVec(i, f(x))
	}
}

func VecAddEach(v *mat64.Vector, x float64) {
	VecApply(v, func(a float64) float64 {
		return a + x
	})
}

func VecClone(v *mat64.Vector) *mat64.Vector {
	ret := mat64.NewVector(v.Len(), nil)
	ret.CloneVec(v)
	return ret
}

func ErrorRate(a, b float64) float64 {
	return math.Abs(math.Abs(a)/math.Abs(b) - 1.0)
}
func NealyEqual(a, b float64) bool {
	return ErrorRate(a, b) < 0.001
}

func Dump(m mat64.Matrix) {
	fmt.Printf("%v\n",
		mat64.Formatted(m, mat64.Prefix(" "), mat64.Excerpt(3)))
}

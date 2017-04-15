package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

type DiagMatrix struct {
	v *mat64.Vector
}

var _ mat64.Matrix = &DiagMatrix{}

func NewDiagMatrix(n int, x []float64) *DiagMatrix {
	return &DiagMatrix{mat64.NewVector(n, x)}
}
func NewDiagMatrixFromVec(v *mat64.Vector) *DiagMatrix {
	return &DiagMatrix{v}
}

func (m *DiagMatrix) Dims() (r, c int) {
	n := m.v.Len()
	return n, n
}
func (m *DiagMatrix) At(i, j int) float64 {
	if i == j {
		return m.v.At(i, 0)
	} else {
		return 0
	}
}
func (m *DiagMatrix) T() mat64.Matrix {
	return m
}

func CrossEntropyError(y, t mat64.Matrix) float64 {
	const delta = 1e-7
	tmp := mat64.DenseCopyOf(y)
	tmp.Apply(func(i, j int, x float64) float64 {
		return math.Log(x+delta) * t.At(i, j)
	}, tmp)
	return -mat64.Sum(tmp)
}

func SoftMaxV(v *mat64.Vector) *mat64.Vector {
	ret := VecClone(v)

	max := mat64.Max(ret)
	VecAddEach(ret, -max)
	VecApply(ret, math.Exp)
	denom := 1.0 / mat64.Sum(ret)
	ret.ScaleVec(denom, ret)
	return ret
}

func SoftMax(m mat64.Matrix) mat64.Matrix {
	max := mat64.Max(m)

	var ret mat64.Dense
	ret.Apply(func(i, j int, x float64) float64 {
		return math.Exp(x - max)
	}, m)

	NormalizeEachRow(&ret)
	return &ret
}

func NormalizeEachRow(m *mat64.Dense) {
	v := SumRows(m, nil)
	VecApply(v, func(x float64) float64 {
		return 1 / x
	})
	k := NewDiagMatrixFromVec(v)
	m.Mul(k, m)
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

func Sum(s []float64) float64 {
	ret := 0.0
	for _, x := range s {
		ret += x
	}
	return ret
}

func SumCols(m mat64.Matrix, to *mat64.Vector) *mat64.Vector {
	r, c := m.Dims()
	if to == nil {
		to = mat64.NewVector(c, nil)
	}
	buf := make([]float64, r)
	for j := 0; j < c; j++ {
		to.SetVec(j, Sum(mat64.Col(buf, j, m)))
	}
	return to
}
func SumRows(m mat64.Matrix, to *mat64.Vector) *mat64.Vector {
	r, c := m.Dims()
	if to == nil {
		to = mat64.NewVector(r, nil)
	}
	buf := make([]float64, c)
	for i := 0; i < r; i++ {
		to.SetVec(i, Sum(mat64.Row(buf, i, m)))
	}
	return to
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

func Dump_(m mat64.Matrix, header string, n int) {
	fmt.Printf("%s %v\n",
		header,
		mat64.Formatted(m, mat64.Prefix(" "), mat64.Excerpt(n)))
}

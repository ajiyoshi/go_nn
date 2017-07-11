package gocnn

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
	"math"
)

type DiagMatrix struct {
	v *mat64.Vector
}

var (
	_ mat64.Matrix  = &DiagMatrix{}
	_ mat64.Mutable = &SubMutable{}
	_ mat64.Mutable = &ZeroPadMutable{}
	_ mat64.Mutable = &ImageMatrix{}
)

func MatFlatten(m mat64.Matrix) []float64 {
	r, c := m.Dims()
	ret := make([]float64, 0, r*c)
	for j := 0; j < r; j++ {
		ret = append(ret, mat64.Row(nil, j, m)...)
	}
	return ret
}

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
	const delta = 1e-6
	var tmp mat64.Dense
	tmp.Apply(func(i, j int, x float64) float64 {
		return math.Log(x+delta) * t.At(i, j)
	}, y)
	r, c := y.Dims()
	if c == 1 {
		return -mat64.Sum(&tmp)
	}
	return -mat64.Sum(&tmp) / float64(r)
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
	max := SumRows(m, nil)

	var ret mat64.Dense
	ret.Apply(func(i, j int, x float64) float64 {
		return math.Exp(x - max.At(i, 0))
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

func Argmax(v []float64) int {
	max := math.Inf(-1)
	ret := 0
	for i, x := range v {
		if x > max {
			max = x
			ret = i
		}
	}
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
	fmt.Printf("%s %.2g\n",
		header,
		mat64.Formatted(m, mat64.Prefix(" "), mat64.Excerpt(n)))
}

func Summary(m mat64.Matrix) string {
	r, c := m.Dims()
	n := float64(r * c)
	ave := mat64.Sum(m) / n
	max := math.Inf(-1)
	min := math.Inf(0)
	ss := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := m.At(i, j)
			ss += x * x
			if x > max {
				max = x
			}
			if x < min {
				min = x
			}
		}
	}
	sigma := ave*ave - ss/n

	return fmt.Sprintf("(%d, %d) max:%.2g min:%.2g ave:%.2g sigma:%.2g", r, c, max, min, ave, sigma)
}

func ZeroPad(ms []mat64.Mutable, n int) []mat64.Mutable {
	ret := make([]mat64.Mutable, len(ms))
	for i, m := range ms {
		ret[i] = NewZeroPadMutable(m, n)
	}
	return ret
}

type SubMutable struct {
	m          mat64.Mutable
	i, j, r, c int
}

func NewSubMutable(m mat64.Mutable, i, j, r, c int) *SubMutable {
	return &SubMutable{
		m: m,
		i: i,
		j: j,
		r: r,
		c: c,
	}
}
func (m *SubMutable) At(i, j int) float64 {
	return m.m.At(m.i+i, m.j+j)
}
func (m *SubMutable) Set(i, j int, x float64) {
	m.m.Set(m.i+i, m.j+j, x)
}
func (m *SubMutable) Dims() (int, int) {
	return m.r, m.c
}
func (m *SubMutable) T() mat64.Matrix {
	return &TransposeMutable{m}
}

type ZeroPadMutable struct {
	m mat64.Mutable
	n int
}

func NewZeroPadMutable(m mat64.Mutable, n int) *ZeroPadMutable {
	return &ZeroPadMutable{
		m: m,
		n: n,
	}
}
func (m *ZeroPadMutable) At(i, j int) float64 {
	i -= m.n
	j -= m.n
	r, c := m.m.Dims()
	if i < 0 || j < 0 || i >= r || j >= c {
		return 0
	}
	return m.m.At(i, j)
}
func (m *ZeroPadMutable) Set(i, j int, x float64) {
	i -= m.n
	j -= m.n
	r, c := m.m.Dims()
	if i < 0 || j < 0 || i >= r || j >= c {
		return
	}
	m.m.Set(i, j, x)
}
func (m *ZeroPadMutable) Dims() (int, int) {
	r, c := m.m.Dims()
	return r + m.n*2, c + m.n*2
}
func (m *ZeroPadMutable) T() mat64.Matrix {
	return &TransposeMutable{m}
}

type ImageMatrix struct {
	img   ImageStrage
	n, ch int
}

func (m *ImageMatrix) At(i, j int) float64 {
	return m.img.Get(m.n, m.ch, i, j)
}
func (m *ImageMatrix) Set(i, j int, x float64) {
	m.img.Set(m.n, m.ch, i, j, x)
}
func (m *ImageMatrix) Dims() (int, int) {
	shape := m.img.Shape()
	return shape.row, shape.col
}
func (m *ImageMatrix) T() mat64.Matrix {
	return &TransposeMutable{m}
}

type TransposeMutable struct {
	m mat64.Mutable
}

func (m *TransposeMutable) At(i, j int) float64 {
	return m.m.At(j, i)
}
func (m *TransposeMutable) Set(i, j int, x float64) {
	m.m.Set(j, i, x)
}
func (m *TransposeMutable) Dims() (int, int) {
	r, c := m.m.Dims()
	return c, r
}
func (m *TransposeMutable) T() mat64.Matrix {
	return &TransposeMutable{m}
}

func MutableApply(m mat64.Mutable, f func(i, j int, x float64) float64) {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, f(i, j, m.At(i, j)))
		}
	}
}

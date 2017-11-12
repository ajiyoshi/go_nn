package nd

import (
	"github.com/ajiyoshi/gocnn/matrix"
	mat "github.com/gonum/matrix/mat64"
)

type ndArrayMatrix struct {
	coef  []int
	array Array
	row   int
	col   int
	buf   []int
}

var (
	_ mat.Matrix  = (*ndArrayMatrix)(nil)
	_ mat.Mutable = (*ndArrayMatrix)(nil)
)

func NewMatrix(row, col int, x Array) *mat.Dense {
	buf := make([]float64, row*col)

	ptr := 0
	for i := x.Iterator(); i.OK(); i.Next() {
		buf[ptr] = x.Get(i.Index()...)
		ptr++
	}
	return mat.NewDense(row, col, buf)
}
func (m *ndArrayMatrix) index(i, j int) []int {
	n := i*m.col + j
	s := m.array.Shape()
	for x, c := range m.coef {
		m.buf[x] = (n / c) % s[x]
	}
	return m.buf
}
func (m *ndArrayMatrix) At(i, j int) float64 {
	return m.array.Get(m.index(i, j)...)
}
func (m *ndArrayMatrix) Set(i, j int, x float64) {
	m.array.Set(x, m.index(i, j)...)
}
func (m *ndArrayMatrix) Dims() (r int, c int) {
	return m.row, m.col
}
func (m *ndArrayMatrix) T() mat.Matrix {
	return matrix.NewTransposeMutable(m)
}

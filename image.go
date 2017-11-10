package gocnn

import (
	"bytes"
	"fmt"

	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/nd"
)

type ImageStrage interface {
	Get(n, ch, r, c int) float64
	Set(n, ch, r, c int, v float64)
	Channels(n int) []mat.Mutable
	Shape() ImageShape
	Matrix() mat.Matrix
	String() string
	Equal(ImageStrage) bool
	Transpose(is ...int) ImageStrage
	ToMatrix(row, col int) mat.Matrix
	Size() int
}

type SimpleStrage struct {
	data nd.Array
}

var (
	_ ImageStrage = &SimpleStrage{}
)

type ImageShape struct {
	n   int
	ch  int
	col int
	row int
}

func NewEmptyStrage(s *ImageShape) *SimpleStrage {
	data := make([]float64, s.n*s.ch*s.row*s.col)
	return NewImages(*s, data)
}

func NewImages(s ImageShape, data []float64) *SimpleStrage {
	array := nd.NewArray(nd.NewShape(s.n, s.ch, s.row, s.col), data)
	return NewSimpleStrage(array)
}

func NewReshaped(s []int, m mat.Matrix) *SimpleStrage {
	data := DumpMatrix(m)
	return NewSimpleStrage(nd.NewArray(s, data))
}

func DumpMatrix(m mat.Matrix) []float64 {
	row, col := m.Dims()
	buf := make([]float64, col)

	data := make([]float64, 0, row*col)
	for i := 0; i < row; i++ {
		mat.Row(buf, i, m)
		data = append(data, buf...)
	}
	return data
}

func NewSimpleStrage(a nd.Array) *SimpleStrage {
	return &SimpleStrage{a}
}

func (img *SimpleStrage) Equal(that ImageStrage) bool {
	return mat.EqualApprox(img.Matrix(), that.Matrix(), 0.001)
}
func (img *SimpleStrage) Shape() ImageShape {
	s := img.data.Shape()
	return ImageShape{n: s[0], ch: s[1], row: s[2], col: s[3]}
}

func (img *SimpleStrage) Get(n, ch, r, c int) float64 {
	return img.data.Get(n, ch, r, c)
}
func (img *SimpleStrage) Set(n, ch, r, c int, v float64) {
	img.data.Set(v, n, ch, r, c)
}
func (img *SimpleStrage) Size() int {
	return img.data.Shape().Size()
}

func (img *SimpleStrage) String() string {
	var buf bytes.Buffer
	for n := 0; n < img.Shape().n; n++ {
		chs := img.Channels(n)
		buf.WriteString("{\n")
		for _, m := range chs {
			buf.WriteString(fmt.Sprintf("%g\n", mat.Formatted(m)))
		}
		buf.WriteString("},\n")
	}
	return buf.String()
}

func (img *SimpleStrage) Channels(n int) []mat.Mutable {
	shape := img.Shape()
	ret := make([]mat.Mutable, shape.ch)
	chs := img.data.Segment(n)
	for i := 0; i < shape.ch; i++ {
		ret[i] = chs.Segment(i).AsMatrix(shape.row, shape.col)
	}
	return ret
}
func (img *SimpleStrage) Matrix() mat.Matrix {
	s := img.Shape()
	return img.ToMatrix(s.n, s.ch*s.col*s.row)
}
func (img *SimpleStrage) ToMatrix(row, col int) mat.Matrix {
	return img.data.AsMatrix(row, col)
}

func (img *SimpleStrage) Transpose(is ...int) ImageStrage {
	return NewSimpleStrage(img.data.Transpose(is...))
}

// (shape.n * outRow * outCol, shape.ch * filterRow * filterCol)
func Im2col(is ImageStrage, filterR, filterC, stride, pad int) *mat.Dense {
	shape := is.Shape()
	outR := (shape.row+2*pad-filterR)/stride + 1
	outC := (shape.col+2*pad-filterC)/stride + 1
	rows := shape.n * outR * outC
	cols := shape.ch * filterR * filterC

	x := 0
	ret := mat.NewDense(rows, cols, nil)
	for n := 0; n < shape.n; n++ {
		ms := matrix.ZeroPad(is.Channels(n), pad)
		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				buf := make([]float64, 0, cols)
				for _, m := range ms {
					s := matrix.NewSubMutable(m, i*stride, j*stride, filterR, filterC)
					buf = append(buf, matrix.MatFlatten(s)...)
				}
				ret.SetRow(x, buf)
				x++
			}
		}
	}

	return ret
}

func Col2im(m mat.Matrix, shape ImageShape, filterR, filterC, stride, pad int) ImageStrage {
	outR := (shape.row+2*pad-filterR)/stride + 1
	outC := (shape.col+2*pad-filterC)/stride + 1

	ret := NewEmptyStrage(&shape)
	x := 0
	for n := 0; n < shape.n; n++ {
		ms := matrix.ZeroPad(ret.Channels(n), pad)

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				row := mat.Row(nil, x, m)
				x++

				for ch, m := range ms {
					offset := ch * filterR * filterC
					len := filterR * filterC
					filter := mat.NewDense(filterR, filterC, row[offset:offset+len])
					s := matrix.NewSubMutable(m, i*stride, j*stride, filterR, filterC)
					matrix.MutableApply(s, func(i, j int, x float64) float64 {
						return x + filter.At(i, j)
					})
				}
			}
		}
	}

	return ret
}

var (
	_ mat.Matrix = &reshapedMatrix{}
)

type reshapedMatrix struct {
	row, col int
	origin   mat.Mutable
}

func ReshapeMatrix(row, col int, m mat.Mutable) *reshapedMatrix {
	r, c := m.Dims()
	if row < 0 {
		row = r * c / col
	}
	if col < 0 {
		col = r * c / row
	}

	if row*col != r*c {
		panic("matrix size should be equal")
	}
	return &reshapedMatrix{
		row:    row,
		col:    col,
		origin: m,
	}
}
func (m *reshapedMatrix) index(i, j int) (r, c int) {
	_, col := m.origin.Dims()
	n := i*m.col + j
	return n / col, n % col
}
func (m *reshapedMatrix) At(i, j int) float64 {
	row, col := m.index(i, j)
	return m.origin.At(row, col)
}
func (m *reshapedMatrix) Set(i, j int, v float64) {
	row, col := m.index(i, j)
	m.origin.Set(row, col, v)
}
func (m *reshapedMatrix) Dims() (int, int) {
	return m.row, m.col
}
func (m *reshapedMatrix) T() mat.Matrix {
	return matrix.NewTransposeMutable(m)
}

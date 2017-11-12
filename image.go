package gocnn

import (
	"bytes"
	"fmt"

	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/nd"
)

type Image interface {
	Get(n, ch, r, c int) float64
	Set(n, ch, r, c int, v float64)
	Channels(n int) []mat.Mutable
	Shape() *Shape
	String() string
	Equal(Image) bool
	Transpose(is ...int) Image
	Matrix() mat.Matrix
	ToMatrix(row, col int) mat.Matrix
	ToArray() nd.Array
	Size() int
}

type ArrayImage struct {
	data nd.Array
}

var (
	_ Image = (*ArrayImage)(nil)
)

type Shape struct {
	N   int
	Ch  int
	Row int
	Col int
}

func (s *Shape) Size() int {
	return s.N * s.Ch * s.Row * s.Col
}

func NewEmptyStrage(s *Shape) *ArrayImage {
	data := make([]float64, s.Size())
	return NewImages(s, data)
}

func NewImages(s *Shape, data []float64) *ArrayImage {
	array := nd.NewArray(nd.NewShape(s.N, s.Ch, s.Row, s.Col), data)
	return NewArrayImage(array)
}

func NewReshaped(s *Shape, m mat.Matrix) *ArrayImage {
	r, c := m.Dims()
	if s.Size() != r*c {
		panic("size should be equal")
	}
	data := DumpMatrix(m)
	return NewImages(s, data)
}

func NewRandomImage(s *Shape, weight float64) *ArrayImage {
	m := matrix.RandamDense(1, s.Size())
	m.Scale(weight, m)
	return NewReshaped(s, m)
}

func NewShape(n, ch, row, col int) *Shape {
	return &Shape{N: n, Ch: ch, Row: row, Col: col}
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

func NewArrayImage(a nd.Array) *ArrayImage {
	return &ArrayImage{a}
}

func (img *ArrayImage) Get(n, ch, r, c int) float64 {
	return img.data.Get(n, ch, r, c)
}
func (img *ArrayImage) Set(n, ch, r, c int, v float64) {
	img.data.Set(v, n, ch, r, c)
}
func (img *ArrayImage) Equal(that Image) bool {
	return img.data.Equals(that.ToArray())
}
func (img *ArrayImage) Shape() *Shape {
	s := img.data.Shape()
	return &Shape{N: s[0], Ch: s[1], Row: s[2], Col: s[3]}
}

func (img *ArrayImage) Size() int {
	return img.data.Shape().Size()
}

func (img *ArrayImage) String() string {
	var buf bytes.Buffer
	for n := 0; n < img.Shape().N; n++ {
		chs := img.Channels(n)
		buf.WriteString("{\n")
		for _, m := range chs {
			buf.WriteString(fmt.Sprintf("%g\n", mat.Formatted(m)))
		}
		buf.WriteString("},\n")
	}
	return buf.String()
}

func (img *ArrayImage) Channels(n int) []mat.Mutable {
	shape := img.Shape()
	ret := make([]mat.Mutable, shape.Ch)
	for ch := 0; ch < shape.Ch; ch++ {
		ret[ch] = img.ChannelMatrix(n, ch)
	}
	return ret
}
func (img *ArrayImage) Matrix() mat.Matrix {
	s := img.Shape()
	return img.ToMatrix(s.N, s.Ch*s.Col*s.Row)
}
func (img *ArrayImage) ToMatrix(row, col int) mat.Matrix {
	return img.data.AsMatrix(row, col)
}
func (img *ArrayImage) ToArray() nd.Array {
	return img.data
}

func (img *ArrayImage) Transpose(is ...int) Image {
	return NewArrayImage(img.data.Transpose(is...))
}

// (shape.n * outRow * outCol, shape.ch * filterRow * filterCol)
func Im2col(is Image, filterR, filterC, stride, pad int) *mat.Dense {
	shape := is.Shape()
	outR := (shape.Row+2*pad-filterR)/stride + 1
	outC := (shape.Col+2*pad-filterC)/stride + 1
	rows := shape.N * outR * outC
	cols := shape.Ch * filterR * filterC

	x := 0
	ret := mat.NewDense(rows, cols, nil)
	for n := 0; n < shape.N; n++ {
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

func Col2im(m mat.Matrix, shape *Shape, filterR, filterC, stride, pad int) Image {
	outR := (shape.Row+2*pad-filterR)/stride + 1
	outC := (shape.Col+2*pad-filterC)/stride + 1

	ret := NewEmptyStrage(shape)
	x := 0
	for n := 0; n < shape.N; n++ {
		ms := matrix.ZeroPad(ret.Channels(n), pad)

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				row := mat.Row(nil, x, m)
				x++

				for ch, mch := range ms {
					offset := ch * filterR * filterC
					length := filterR * filterC
					filter := mat.NewDense(filterR, filterC, row[offset:offset+length])
					s := matrix.NewSubMutable(mch, i*stride, j*stride, filterR, filterC)
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
	_ mat.Matrix = &ChannelMatrix{}
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

type ChannelMatrix struct {
	img   Image
	n, ch int
}

func (img *ArrayImage) ChannelMatrix(n, ch int) *ChannelMatrix {
	return &ChannelMatrix{
		img: img,
		n:   n,
		ch:  ch,
	}
}

func (m *ChannelMatrix) At(i, j int) float64 {
	return m.img.Get(m.n, m.ch, i, j)
}
func (m *ChannelMatrix) Set(i, j int, x float64) {
	m.img.Set(m.n, m.ch, i, j, x)
}
func (m *ChannelMatrix) Dims() (int, int) {
	shape := m.img.Shape()
	return shape.Row, shape.Col
}
func (m *ChannelMatrix) T() mat.Matrix {
	return matrix.NewTransposeMutable(m)
}

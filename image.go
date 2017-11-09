package gocnn

import (
	"bytes"
	"fmt"

	"github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/nd"
)

type ImageStrage interface {
	Get(n, ch, r, c int) float64
	Set(n, ch, r, c int, v float64)
	Channels(n int) []mat64.Mutable
	Shape() ImageShape
	Matrix() mat64.Matrix
	String() string
	Equal(ImageStrage) bool
	Transpose(is ...int) ImageStrage
	ToMatrix(row, col int) mat64.Matrix
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

func NewReshaped(s ImageShape, m mat64.Matrix) *SimpleStrage {
	data := DumpMatrix(m)
	return NewSimpleStrage(nd.NewArray(nd.NewShape(s.n, s.ch, s.row, s.col), data))
}

func DumpMatrix(m mat64.Matrix) []float64 {
	row, col := m.Dims()
	buf := make([]float64, col)

	data := make([]float64, 0, row*col)
	for i := 0; i < row; i++ {
		mat64.Row(buf, i, m)
		data = append(data, buf...)
	}
	return data
}

func NewSimpleStrage(a nd.Array) *SimpleStrage {
	return &SimpleStrage{a}
}

func (img *SimpleStrage) Equal(that ImageStrage) bool {
	return mat64.EqualApprox(img.Matrix(), that.Matrix(), 0.001)
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

func (img *SimpleStrage) String() string {
	var buf bytes.Buffer
	for n := 0; n < img.Shape().n; n++ {
		chs := img.Channels(n)
		buf.WriteString("{\n")
		for _, m := range chs {
			buf.WriteString(fmt.Sprintf("%g\n", mat64.Formatted(m)))
		}
		buf.WriteString("},\n")
	}
	return buf.String()
}

func (img *SimpleStrage) Channels(n int) []mat64.Mutable {
	shape := img.Shape()
	ret := make([]mat64.Mutable, shape.ch)
	for i := 0; i < shape.ch; i++ {
		ret[i] = img.ChannelMatrix(n, i)
	}
	return ret
}
func (img *SimpleStrage) ChannelMatrix(n, ch int) *ChannelMatrix {
	return &ChannelMatrix{
		img: img,
		n:   n,
		ch:  ch,
	}
}

func (img *SimpleStrage) Matrix() mat64.Matrix {
	s := img.Shape()
	return img.ToMatrix(s.n, s.ch*s.col*s.row)
}
func (img *SimpleStrage) ToMatrix(row, col int) mat64.Matrix {
	ret := mat64.NewDense(row, col, make([]float64, col*row))
	itr := img.data.Iterator()
	for j := 0; j < row; j++ {
		for i := 0; i < col; i++ {
			ret.Set(j, i, itr.Value())
			itr.Next()
		}
	}
	return ret
}

func (img *SimpleStrage) Transpose(is ...int) ImageStrage {
	return NewSimpleStrage(img.data.Transpose(is...))
}

// (shape.n * outRow * outCol, shape.ch * filterRow * filterCol)
func Im2col(is ImageStrage, filterR, filterC, stride, pad int) mat64.Matrix {
	shape := is.Shape()
	outR := (shape.row+2*pad-filterR)/stride + 1
	outC := (shape.col+2*pad-filterC)/stride + 1
	rows := shape.n * outR * outC
	cols := shape.ch * filterR * filterC

	x := 0
	ret := mat64.NewDense(rows, cols, nil)
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

func Col2im(m mat64.Matrix, shape *ImageShape, filterR, filterC, stride, pad int) ImageStrage {
	outR := (shape.row+2*pad-filterR)/stride + 1
	outC := (shape.col+2*pad-filterC)/stride + 1

	ret := NewEmptyStrage(shape)
	x := 0
	for n := 0; n < shape.n; n++ {
		ms := matrix.ZeroPad(ret.Channels(n), pad)

		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				row := mat64.Row(nil, x, m)
				x++

				for ch, m := range ms {
					offset := ch * filterR * filterC
					len := filterR * filterC
					filter := mat64.NewDense(filterR, filterC, row[offset:offset+len])
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
	_ mat64.Mutable = &ChannelMatrix{}
)

type ChannelMatrix struct {
	img   ImageStrage
	n, ch int
}

func (m *ChannelMatrix) At(i, j int) float64 {
	return m.img.Get(m.n, m.ch, i, j)
}
func (m *ChannelMatrix) Set(i, j int, x float64) {
	m.img.Set(m.n, m.ch, i, j, x)
}
func (m *ChannelMatrix) Dims() (int, int) {
	shape := m.img.Shape()
	return shape.row, shape.col
}
func (m *ChannelMatrix) T() mat64.Matrix {
	return matrix.NewTransposeMutable(m)
}

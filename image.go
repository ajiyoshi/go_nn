package gocnn

import (
	"github.com/gonum/matrix/mat64"
	"reflect"

	"github.com/ajiyoshi/gocnn/matrix"
)

type ImageStrage interface {
	Get(n, ch, r, c int) float64
	Set(n, ch, r, c int, v float64)
	Channels(n int) []mat64.Mutable
	Shape() ImageShape
	Equal(ImageStrage) bool
}

type SimpleStrage struct {
	shape ImageShape
	data  []float64
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

func NewEmptyStrage(shape *ImageShape) *SimpleStrage {
	return &SimpleStrage{
		shape: *shape,
		data:  make([]float64, shape.n*shape.ch*shape.row*shape.col),
	}
}

func NewImages(shape ImageShape, data []float64) *SimpleStrage {
	return &SimpleStrage{
		shape: shape,
		data:  data,
	}
}

func (img *SimpleStrage) Equal(that ImageStrage) bool {
	shape := that.Shape()
	if !reflect.DeepEqual(&img.shape, &shape) {
		return false
	}
	for n := 0; n < shape.n; n++ {
		for ch := 0; ch < shape.ch; ch++ {
			for row := 0; row < shape.row; row++ {
				for col := 0; col < shape.col; col++ {
					if img.Get(n, ch, row, col) != that.Get(n, ch, row, col) {
						return false
					}
				}
			}
		}
	}

	return true
}
func (img *SimpleStrage) Shape() ImageShape {
	return img.shape
}

func (img *SimpleStrage) Get(n, ch, r, c int) float64 {
	s := img.Shape()
	index := n*(s.ch*s.row*s.col) + ch*s.row*s.col + r*s.col + c
	return img.data[index]
}
func (img *SimpleStrage) Set(n, ch, r, c int, v float64) {
	s := img.Shape()
	index := n*(s.ch*s.row*s.col) + ch*s.row*s.col + r*s.col + c
	//fmt.Printf("(%d, %d, %d, %d)[%d] %f <- %f\n", n, ch, r, c, index, img.data[index], v)
	img.data[index] = v
}

func (img *SimpleStrage) Channels(n int) []mat64.Mutable {
	ret := make([]mat64.Mutable, img.shape.ch)
	for i := 0; i < img.shape.ch; i++ {
		ret[i] = img.AsMatrix(n, i)
	}
	return ret
}
func (img *SimpleStrage) AsMatrix(n, ch int) *ImageMatrix {
	return &ImageMatrix{
		img: img,
		n:   n,
		ch:  ch,
	}
}

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
	_ mat64.Matrix  = &ImageMatrix{}
	_ mat64.Mutable = &ImageMatrix{}
)

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
	return matrix.NewTransposeMutable(m)
}

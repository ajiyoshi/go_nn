package main

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

type ImageStrage interface {
	Get(n, ch, r, c int) float64
	Channels(n int) []mat64.Matrix
	Num() int
	Col() int
	Row() int
	Chan() int
}

type SimpleStrage struct {
	N    int
	Ch   int
	Col  int
	Row  int
	data []float64
}

func NewImages(n, ch, r, c int, data []float64) *SimpleStrage {
	return &SimpleStrage{
		N:    n,
		Ch:   ch,
		Row:  r,
		Col:  c,
		data: data,
	}
}

func (img *SimpleStrage) Get(n, ch, r, c int) float64 {
	index := n*(img.Ch*img.Row*img.Col) + ch*img.Row*img.Col + r*img.Col + c
	return img.data[index]
}

func (img *SimpleStrage) Channels(n int) []mat64.Matrix {
	ret := make([]mat64.Matrix, img.Ch)
	for i := 0; i < img.Ch; i++ {
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

type ImageMatrix struct {
	img   *SimpleStrage
	n, ch int
}

var _ mat64.Matrix = &ImageMatrix{}

func (m *ImageMatrix) At(i, j int) float64 {
	return m.img.Get(m.n, m.ch, i, j)
}
func (m *ImageMatrix) Dims() (int, int) {
	return m.img.Row, m.img.Col
}
func (m *ImageMatrix) T() mat64.Matrix {
	return NewSubMatrix(m, 0, 0, m.img.Row, m.img.Col).T()
}

func ZeroPad(ms []mat64.Matrix, n int) []mat64.Matrix {
	ret := make([]mat64.Matrix, len(ms))
	for i, m := range ms {
		ret[i] = NewZeroPadMatrix(m, n)
	}
	return ret
}

func Im2col(is *SimpleStrage, filterR, filterC, stride, pad int) mat64.Matrix {
	outR := (is.Row+2*pad-filterR)/stride + 1
	outC := (is.Col+2*pad-filterC)/stride + 1
	rows := is.N * outR * outC
	cols := is.Ch * filterR * filterC

	x := 0
	ret := mat64.NewDense(rows, cols, nil)
	for n := 0; n < is.N; n++ {
		ms := ZeroPad(is.Channels(n), pad)
		for i := 0; i < outR; i++ {
			for j := 0; j < outC; j++ {
				buf := make([]float64, 0, cols)
				for _, m := range ms {
					s := NewSubMatrix(m, i*stride, j*stride, filterR, filterC)
					fmt.Println(mat64.Formatted(s))
					buf = append(buf, MatFlatten(s)...)
				}
				ret.SetRow(x, buf)
				x++
			}
		}
	}

	return ret
}

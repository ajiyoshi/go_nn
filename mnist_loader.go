package main

import (
	"bytes"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"io"
	"math/rand"
)

type TrainBuffer struct {
	rows  int
	x_col int
	t_col int
	x     []float64
	t     []float64
}

func NewTrainBuffer(rows, x_col, t_col int) *TrainBuffer {
	return &TrainBuffer{
		rows:  rows,
		x_col: x_col,
		t_col: t_col,
		x:     make([]float64, rows*x_col),
		t:     make([]float64, rows*t_col),
	}
}
func (buf *TrainBuffer) LoadX(i int, x []byte) error {
	if len(x) != buf.x_col {
		return fmt.Errorf("bad size of data expect:%d but got %d", buf.x_col, len(x))
	}
	offset := i * buf.x_col
	LoadVec(x, buf.x[offset:])
	return nil
}
func (buf *TrainBuffer) LoadT(i int, t byte) {
	offset := i * buf.t_col
	copy(buf.t[offset:], labels[t])
}

func (buf *TrainBuffer) Load(m *Mnist, at []int) {
	rows := len(at)
	for i := 0; i < rows; i++ {
		n := at[i]
		x, t := m.At(n)
		buf.LoadX(i, x)
		buf.LoadT(i, t)
	}
}

func (buf *TrainBuffer) Dump(w io.Writer) {
	for i := 0; i < buf.rows; i++ {
		offset := buf.x_col * i
		fmt.Fprintf(w, "%s\n", XToString(buf.x[offset:]))
	}
}

func (buf *TrainBuffer) Bake() (x, t mat64.Matrix) {
	x = mat64.NewDense(buf.rows, buf.x_col, buf.x)
	t = mat64.NewDense(buf.rows, buf.t_col, buf.t)
	return x, t
}

func DumpX(w io.Writer, x, t mat64.Matrix, i int) {
	xbuf := mat64.Row(nil, i, x)
	tbuf := mat64.Row(nil, i, t)
	fmt.Fprintf(w, "%d:\n%s\n", Label(tbuf), XToString(xbuf))
}

func Label(v []float64) int {
	vec := mat64.NewVector(len(v), v)
	return ArgmaxV(vec)
}

func GetX(data []float64, x, y int) float64 {
	return data[x+y*28]
}
func AddHoge(data []float64, v float64, x, y int) {
	data[(x/4)+(y/4)*7] += v
}
func GetHoge(data []float64, x, y int) float64 {
	return data[x+7*y]
}

func XToString(data []float64) string {
	w := bytes.NewBuffer(make([]byte, 0, 50))
	buf := make([]float64, 49)

	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			s := GetX(data, x, y)
			AddHoge(buf, s, x, y)
		}
	}
	for y := 0; y < 7; y++ {
		for x := 0; x < 7; x++ {
			if GetHoge(buf, x, y) > 5 {
				w.WriteString("x")
			} else {
				w.WriteString(".")
			}
		}
		w.WriteString("\n")
	}
	return w.String()
}

func LoadVec(raw []byte, buf []float64) {
	for i, v := range raw {
		buf[i] = float64(v) / 255.0
	}
}

var labels [][]float64 = [][]float64{
	[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
}

func LoadLabel(label byte) *mat64.Vector {
	return mat64.NewVector(10, labels[label])
}

func seq(x, n int) []int {
	ret := make([]int, n)
	for i := 0; i < n; i++ {
		ret[i] = x + i
	}
	return ret
}

func randamSeq(n, max int) []int {
	ret := make([]int, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.Intn(max)
	}
	return ret
}

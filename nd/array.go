package nd

import (
	"fmt"
	"strings"

	mat "github.com/gonum/matrix/mat64"
)

type Array interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() Shape
	Segment(i int) Array
	Transpose(is ...int) Array
	String() string
	DeepEqual(Array) bool
	AsMatrix(row, col int) mat.Mutable
}

type Shape []int

type Indexer interface {
	At(is ...int) int
}

var (
	_ Array = (*ndArray)(nil)

	_ Indexer = (*NormalIndexer)(nil)
	_ Indexer = (*TransposeIndexer)(nil)
	_ Indexer = (*SubIndexer)(nil)
)

type ndArray struct {
	data  []float64
	shape Shape
	index Indexer
}

func NewArray(s Shape, data []float64) *ndArray {
	return &ndArray{
		shape: s,
		data:  data,
		index: &NormalIndexer{s},
	}
}
func (x *ndArray) Get(is ...int) float64 {
	i := x.index.At(is...)
	return x.data[i]
}
func (x *ndArray) Set(v float64, is ...int) {
	i := x.index.At(is...)
	x.data[i] = v
}
func (x *ndArray) Shape() Shape {
	return x.shape
}
func (x *ndArray) Segment(i int) Array {
	return &ndArray{
		shape: NewSubShape(x.shape),
		data:  x.data,
		index: &SubIndexer{fixed: i, origin: x.index},
	}
}
func (x *ndArray) Transpose(is ...int) Array {
	return &ndArray{
		shape: NewTrShape(x.shape, is),
		data:  x.data,
		index: &TransposeIndexer{table: is, origin: x.index},
	}
}
func (x *ndArray) String() string {
	s := x.shape
	if len(s) == 1 {
		tmp := make([]string, s[0])
		for i := 0; i < len(tmp); i++ {
			tmp[i] = fmt.Sprintf("%.2f", x.Get(i))
		}
		return fmt.Sprintf("[%s]", strings.Join(tmp, ", "))
	}

	tmp := make([]string, s[0])
	for i := 0; i < s[0]; i++ {
		sub := x.Segment(i)
		tmp[i] = sub.String()
	}
	return fmt.Sprintf("[%s]", strings.Join(tmp, ",\n"))
}
func (x *ndArray) DeepEqual(y Array) bool {
	if !x.Shape().Equals(y.Shape()) {
		return false
	}
	n := x.Shape().Size()
	this, that := x.AsMatrix(1, n), y.AsMatrix(1, n)
	return mat.EqualApprox(this, that, 0.01)
}

func (x *ndArray) AsMatrix(row, col int) mat.Mutable {
	return NewMatrix(row, col, x)
}

func (s Shape) Size() int {
	ret := 1
	for _, x := range s {
		ret *= x
	}
	return ret
}

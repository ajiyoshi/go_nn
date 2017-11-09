package nd

import (
	"fmt"
	"strings"
)

type Array interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() Shape
	Segment(i int) Array
	Transpose(is ...int) Array
	String() string
	DeepEqual(Array) bool
	Iterator() ArrayIterator
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
	return x.String() == y.String()
}
func (x *ndArray) Iterator() ArrayIterator {
	coef := Coefficient(x.Shape())
	max := coef[0]
	return &ndArrayIterator{
		i:     0,
		array: x,
		coef:  coef,
		max:   max,
		buf:   make([]int, len(x.Shape())),
	}
}

type ndArrayIterator struct {
	i     int
	array Array
	max   int
	coef  []int
	buf   []int
}
type ArrayIterator interface {
	OK() bool
	Value() float64
	Index() []int
	Next()
}

func (itr *ndArrayIterator) OK() bool {
	return itr.i < itr.max
}
func (itr *ndArrayIterator) Value() float64 {
	return itr.array.Get(itr.Index()...)
}
func (itr *ndArrayIterator) Next() {
	itr.i++
	for i, c := range itr.coef {
		itr.buf[i] = itr.i / c % itr.array.Shape()[i]
	}
}
func (itr *ndArrayIterator) Index() []int {
	return itr.buf
}

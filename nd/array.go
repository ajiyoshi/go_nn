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
	Equals(Array) bool
	EqualApprox(Array, float64) bool
	AsMatrix(row, col int) *mat.Dense
	Iterator() Iterator

	Scale(float64) Array
	AddSalar(float64) Array
	AddEach(Array) Array
	SubEach(Array) Array
	MulEach(Array) Array
	DivEach(Array) Array

	Map(func(float64) float64) Array
	Clone() Array
}

type Shape []int

type Indexer interface {
	At(is ...int) int
}

type Iterable interface {
	Iterator() Iterator
}

type Iterator interface {
	OK() bool
	Index() []int
	Reset()
	Next()
}

var (
	_ Array = (*ndArray)(nil)

	_ Indexer = (*NormalIndexer)(nil)
	_ Indexer = (*TransposeIndexer)(nil)
	_ Indexer = (*SubIndexer)(nil)

	_ Iterable = (*ndArray)(nil)
	_ Iterable = (Shape)(nil)
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
func Zeros(s Shape) *ndArray {
	return NewArray(s, make([]float64, s.Size()))
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
func (x *ndArray) Equals(y Array) bool {
	return x.EqualApprox(y, 0.001)
}
func (x *ndArray) EqualApprox(y Array, e float64) bool {
	if !x.Shape().Equals(y.Shape()) {
		return false
	}
	n := x.Shape().Size()
	this, that := x.AsMatrix(1, n), y.AsMatrix(1, n)
	return mat.EqualApprox(this, that, e)
}

func (x *ndArray) AsMatrix(row, col int) *mat.Dense {
	return NewMatrix(row, col, x)
}

func (x *ndArray) Iterator() Iterator {
	return x.shape.Iterator()
}

func (x *ndArray) Scale(k float64) Array {
	for i := x.Iterator(); i.OK(); i.Next() {
		index := i.Index()
		val := x.Get(index...)
		x.Set(val*k, index...)
	}
	return x
}
func (x *ndArray) AddSalar(k float64) Array {
	for i := x.Iterator(); i.OK(); i.Next() {
		index := i.Index()
		val := x.Get(index...)
		x.Set(val+k, index...)
	}
	return x
}
func (x *ndArray) AddEach(y Array) Array {
	return x.Each(y, func(a, b float64) float64 {
		return a + b
	})
}
func (x *ndArray) SubEach(y Array) Array {
	return x.Each(y, func(a, b float64) float64 {
		return a - b
	})
}
func (x *ndArray) MulEach(y Array) Array {
	return x.Each(y, func(a, b float64) float64 {
		return a * b
	})
}
func (x *ndArray) DivEach(y Array) Array {
	return x.Each(y, func(a, b float64) float64 {
		return a / b
	})
}
func (x *ndArray) Each(y Array, op func(float64, float64) float64) Array {
	if !x.Shape().Equals(y.Shape()) {
		panic("shape should be same")
	}
	i := x.Shape().Iterator()
	for i.Reset(); i.OK(); i.Next() {
		index := i.Index()
		a, b := x.Get(index...), y.Get(index...)
		x.Set(op(a, b), index...)
	}
	return x
}
func (x *ndArray) Map(f func(float64) float64) Array {
	ret := Zeros(x.shape)
	for i := x.Iterator(); i.OK(); i.Next() {
		index := i.Index()
		v := x.Get(index...)
		ret.Set(f(v), index...)
	}
	return ret
}
func (x *ndArray) Clone() Array {
	ret := Zeros(x.shape)
	for i := x.Iterator(); i.OK(); i.Next() {
		index := i.Index()
		v := x.Get(index...)
		x.Set(v, index...)
	}
	return ret
}

func (s Shape) Size() int {
	ret := 1
	for _, x := range s {
		ret *= x
	}
	return ret
}

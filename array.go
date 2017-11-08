package gocnn

import (
	"fmt"
	"reflect"
	"strings"
)

type NDArray interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() NDShape
	Segment(i int) NDArray
	Transpose(is ...int) NDArray
	String() string
	DeepEqual(NDArray) bool
}

type NDShape []int

type Indexer interface {
	At(is ...int) int
}

var (
	_ NDArray = (*ndArray)(nil)

	_ Indexer = (*NormalIndexer)(nil)
	_ Indexer = (*TransposeIndexer)(nil)
	_ Indexer = (*SubIndexer)(nil)
)

type ndArray struct {
	data  []float64
	shape NDShape
	index Indexer
}

func NewNDArray(s NDShape, data []float64) *ndArray {
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
func (x *ndArray) Shape() NDShape {
	return x.shape
}
func (x *ndArray) Segment(i int) NDArray {
	return &ndArray{
		shape: NewSubShape(x.shape),
		data:  x.data,
		index: &SubIndexer{fixed: i, origin: x.index},
	}
}
func (x *ndArray) Transpose(is ...int) NDArray {
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
			tmp[i] = fmt.Sprintf("%f", x.Get(i))
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
func (x *ndArray) DeepEqual(y NDArray) bool {
	if !x.Shape().Equals(y.Shape()) {
		return false
	}
	return x.String() == y.String()
}

func NewNDShape(dim ...int) NDShape {
	return dim
}
func NewSubShape(origin NDShape) NDShape {
	return origin[1:]
}
func NewTrShape(origin NDShape, table []int) NDShape {
	return shapeConvert(origin, table)
}
func shapeConvert(origin NDShape, table []int) NDShape {
	/*
		shape(2, 3) (1, 0) -> shape(3, 2)
		shape(2, 3, 4) (1, 2, 0) -> shape(3, 4, 2)
	*/
	ret := make([]int, len(origin))
	for i, at := range table {
		ret[i] = origin[at]
	}
	return ret
}

func (x NDShape) Equals(y NDShape) bool {
	return reflect.DeepEqual(x, y)
}

type NormalIndexer struct {
	shape NDShape
}
type TransposeIndexer struct {
	table  []int
	origin Indexer
}
type SubIndexer struct {
	fixed  int
	origin Indexer
}

func (x *NormalIndexer) At(is ...int) int {
	ret := 0
	coef := Coefficient(x.shape)
	for i, x := range is {
		ret += coef[i] * x
	}
	return ret
}

func (x *TransposeIndexer) At(is ...int) int {
	return x.origin.At(indexConvert(is, x.table)...)
}
func indexConvert(origin, table []int) []int {
	/*
		unspeakable spec
			index(1, 2) (1, 0) -> index(2, 1)
			index(0, 0, 1) (1, 2, 0) -> index(1, 0, 0)
	*/
	ret := make([]int, len(origin))
	for i, at := range table {
		ret[at] = origin[i]
	}
	return ret
}

func (x *SubIndexer) At(is ...int) int {
	return x.origin.At(Cons(x.fixed, is)...)
}

func Cons(car int, cdr []int) []int {
	ret := make([]int, len(cdr)+1)
	ret[0], ret = car, append(ret[:1], cdr...)
	return ret
}

func Coefficient(dims []int) []int {
	/*
		[]int{ d0, d1, d2, ... dn }
		-> []int{ (d1*d2*...*dn), (d2*...*dn), ... dn, 1 }
	*/
	ptr, length := dims, len(dims)
	ret := make([]int, length)
	acc := 1
	for i := 0; i < length; i++ {
		ret[length-i-1] = acc
		//末尾を掛ける
		acc *= ptr[len(ptr)-1]
		//末尾を削除
		ptr = ptr[:len(ptr)-1]
	}
	return ret
}

func Convert(table, is []int) []int {
	ret := make([]int, len(is))
	for i, x := range table {
		ret[i] = is[x]
	}
	return ret
}

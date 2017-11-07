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
type NDShape interface {
	Index(is ...int) int
	AsSlice() []int
	Equals(NDShape) bool
}

var (
	_ NDArray = (*ndArray)(nil)
	_ NDShape = (*ndShape)(nil)
	_ NDShape = (*ndSubShape)(nil)
	_ NDShape = (*ndTrShape)(nil)
)

type ndArray struct {
	data  []float64
	shape NDShape
}
type ndShape struct {
	dims []int
}
type ndSubShape struct {
	base  int
	shape NDShape
}
type ndTrShape struct {
	table []int
	shape NDShape
}

func NewNDArray(s NDShape, data []float64) *ndArray {
	return &ndArray{
		shape: s,
		data:  data,
	}
}
func (x *ndArray) Get(is ...int) float64 {
	i := x.shape.Index(is...)
	return x.data[i]
}

func (x *ndArray) Set(v float64, is ...int) {
	i := x.shape.Index(is...)
	x.data[i] = v
}
func (x *ndArray) At(i int) NDArray {
	return nil
}

func (x *ndArray) Shape() NDShape {
	return x.shape
}
func (x *ndArray) Segment(i int) NDArray {
	return &ndArray{
		shape: NewSubShape(x.shape, i),
		data:  x.data,
	}
}
func (x *ndArray) Transpose(is ...int) NDArray {
	return &ndArray{
		shape: NewTrShape(x.shape, is...),
		data:  x.data,
	}
}
func (x *ndArray) String() string {
	s := x.shape.AsSlice()
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
	if !x.shape.Equals(y.Shape()) {
		return false
	}
	return x.String() == y.String()
}

func NewNDShape(ds ...int) *ndShape {
	return &ndShape{
		dims: ds,
	}
}
func (s *ndShape) AsSlice() []int {
	return s.dims
}
func (s *ndShape) Index(is ...int) int {
	ret := 0
	coef := s.Coefficient()
	for i, x := range is {
		ret += coef[i] * x
	}
	return ret
}
func (s *ndShape) Equals(y NDShape) bool {
	return reflect.DeepEqual(s.dims, y.AsSlice())
}
func (s *ndShape) Coefficient() []int {
	/*
		[]int{ (d1*d2*...*dn), (d2*...*dn), ... dn, 1 }
	*/
	buf := s.dims
	length := len(buf)
	ret := make([]int, length)
	acc := 1
	for i := 0; i < length; i++ {
		ret[length-i-1] = acc
		//末尾を掛ける
		acc *= buf[len(buf)-1]
		//末尾を削除
		buf = buf[:len(buf)-1]
	}
	return ret
}

func prod(xs []int) int {
	ret := 1
	for _, x := range xs {
		ret *= x
	}
	return ret
}

func NewSubShape(s NDShape, i int) *ndSubShape {
	slice := s.AsSlice()
	cdr := slice[1:]
	return &ndSubShape{
		base:  i * prod(cdr),
		shape: NewNDShape(cdr...),
	}
}
func (s *ndSubShape) AsSlice() []int {
	return s.shape.AsSlice()
}
func (s *ndSubShape) Index(is ...int) int {
	return s.base + s.shape.Index(is...)
}
func (s *ndSubShape) Equals(y NDShape) bool {
	return reflect.DeepEqual(s.shape.AsSlice(), y.AsSlice())
}

func NewTrShape(s NDShape, is ...int) *ndTrShape {
	if len(s.AsSlice()) != len(is) {
		panic("dimention mismatch")
	}
	return &ndTrShape{
		table: is,
		shape: s,
	}
}
func Convert(table, is []int) []int {
	ret := make([]int, len(is))
	for i, x := range table {
		ret[i] = is[x]
	}
	return ret
}
func (s *ndTrShape) AsSlice() []int {
	return Convert(s.table, s.shape.AsSlice())
}
func (s *ndTrShape) Equals(that NDShape) bool {
	return reflect.DeepEqual(s.shape.AsSlice(), that.AsSlice())
}
func (s *ndTrShape) Index(is ...int) int {
	return s.shape.Index(Convert(s.table, is)...)
}

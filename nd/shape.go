package nd

import (
	"reflect"
)

func NewShape(dim ...int) Shape {
	return dim
}
func NewSubShape(origin Shape) Shape {
	return origin[1:]
}
func NewTrShape(origin Shape, table []int) Shape {
	return shapeConvert(origin, table)
}
func shapeConvert(origin Shape, table []int) Shape {
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

func (x Shape) Equals(y Shape) bool {
	return reflect.DeepEqual(x, y)
}

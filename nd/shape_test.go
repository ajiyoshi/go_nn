package nd

import (
	"testing"
)

func TestTransposedShape(t *testing.T) {
	cases := []struct {
		msg    string
		input  Array
		index  []int
		expect Shape
	}{
		{
			msg:    "(2, 3).Traspose(1, 0)",
			input:  NewArray(NewShape(2, 3), make([]float64, 6)),
			index:  []int{1, 0},
			expect: NewShape(3, 2),
		},
		{
			msg:    "(2, 3, 4).Traspose(1, 2, 0)",
			input:  NewArray(NewShape(2, 3, 4), make([]float64, 24)),
			index:  []int{1, 2, 0},
			expect: NewShape(3, 4, 2),
		},
	}

	for _, c := range cases {
		actual := c.input.Transpose(c.index...).Shape()
		if !c.expect.Equals(actual) {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}

func TestShapeConvert(t *testing.T) {
	cases := []struct {
		msg    string
		shape  Shape
		table  []int
		expect []int
	}{
		{msg: "shape{4, 5, 6} by{1, 2, 0} -> shape{5, 6, 4}",
			shape: []int{4, 5, 6}, table: []int{1, 2, 0}, expect: []int{5, 6, 4}},
		{msg: "shape{4, 5, 6} by{0, 1, 2} -> shape{4, 5, 6}",
			shape: []int{4, 5, 6}, table: []int{0, 1, 2}, expect: []int{4, 5, 6}},
	}
	for _, c := range cases {
		actual := shapeConvert(c.shape, c.table)
		if !actual.Equals(c.expect) {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}

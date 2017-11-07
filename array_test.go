package gocnn

import (
	"testing"
)

func TestNDArray(t *testing.T) {
	a := NewNormalND(NewShapeND(1, 2, 3, 4),
		[]float64{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,

			13, 14, 15, 16,
			17, 18, 19, 20,
			21, 22, 23, 24,
		})

	cases := []struct {
		msg    string
		index  []int
		expect float64
	}{
		{
			msg:    "(0, 0, 0, 0)",
			index:  []int{0, 0, 0, 0},
			expect: 1,
		},
		{
			msg:    "(0, 0, 0, 1)",
			index:  []int{0, 0, 0, 1},
			expect: 2,
		},
		{
			msg:    "(0, 0, 0, 2)",
			index:  []int{0, 0, 0, 2},
			expect: 3,
		},
		{
			msg:    "(0, 0, 0, 3)",
			index:  []int{0, 0, 0, 3},
			expect: 4,
		},

		{
			msg:    "(0, 0, 1, 0)",
			index:  []int{0, 0, 1, 0},
			expect: 5,
		},
		{
			msg:    "(0, 0, 1, 1)",
			index:  []int{0, 0, 1, 1},
			expect: 6,
		},
		{
			msg:    "(0, 0, 1, 2)",
			index:  []int{0, 0, 1, 2},
			expect: 7,
		},
		{
			msg:    "(0, 0, 1, 3)",
			index:  []int{0, 0, 1, 3},
			expect: 8,
		},

		{
			msg:    "(0, 0, 2, 0)",
			index:  []int{0, 0, 2, 0},
			expect: 9,
		},
		{
			msg:    "(0, 0, 2, 1)",
			index:  []int{0, 0, 2, 1},
			expect: 10,
		},
		{
			msg:    "(0, 0, 2, 2)",
			index:  []int{0, 0, 2, 2},
			expect: 11,
		},
		{
			msg:    "(0, 0, 2, 3)",
			index:  []int{0, 0, 2, 3},
			expect: 12,
		},

		{
			msg:    "(0, 1, 0, 0)",
			index:  []int{0, 1, 0, 0},
			expect: 13,
		},
		{
			msg:    "(0, 1, 0, 1)",
			index:  []int{0, 1, 0, 1},
			expect: 14,
		},
		{
			msg:    "(0, 1, 0, 2)",
			index:  []int{0, 1, 0, 2},
			expect: 15,
		},
		{
			msg:    "(0, 1, 0, 3)",
			index:  []int{0, 1, 0, 3},
			expect: 16,
		},

		{
			msg:    "(0, 1, 1, 0)",
			index:  []int{0, 1, 1, 0},
			expect: 17,
		},
		{
			msg:    "(0, 1, 1, 1)",
			index:  []int{0, 1, 1, 1},
			expect: 18,
		},
		{
			msg:    "(0, 1, 1, 2)",
			index:  []int{0, 1, 1, 2},
			expect: 19,
		},
		{
			msg:    "(0, 1, 1, 3)",
			index:  []int{0, 1, 1, 3},
			expect: 20,
		},

		{
			msg:    "(0, 1, 2, 0)",
			index:  []int{0, 1, 2, 0},
			expect: 21,
		},
		{
			msg:    "(0, 1, 2, 1)",
			index:  []int{0, 1, 2, 1},
			expect: 22,
		},
		{
			msg:    "(0, 1, 2, 2)",
			index:  []int{0, 1, 2, 2},
			expect: 23,
		},
		{
			msg:    "(0, 1, 2, 3)",
			index:  []int{0, 1, 2, 3},
			expect: 24,
		},
	}
	for _, c := range cases {
		actual := a.Get(c.index...)
		if c.expect != actual {
			t.Fatalf("(%s) expect %v but got actual %v", c.msg, c.expect, actual)
		}
	}

}
func TestTransposedArray(t *testing.T) {
}

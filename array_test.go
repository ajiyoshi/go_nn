package gocnn

import (
	"testing"
)

func TestNDArrayGet(t *testing.T) {
	a := NewNDArray(NewNDShape(2, 3),
		[]float64{
			1, 2, 3,
			4, 5, 6,
		})

	cases := []struct {
		msg    string
		index  []int
		expect float64
	}{
		{
			msg:    "(0, 0)",
			index:  []int{0, 0},
			expect: 1,
		},
		{
			msg:    "(0, 1)",
			index:  []int{0, 1},
			expect: 2,
		},
		{
			msg:    "(0, 2)",
			index:  []int{0, 2},
			expect: 3,
		},
		{
			msg:    "(1, 0)",
			index:  []int{1, 0},
			expect: 4,
		},
		{
			msg:    "(1, 1)",
			index:  []int{1, 1},
			expect: 5,
		},
		{
			msg:    "(1, 2)",
			index:  []int{1, 2},
			expect: 6,
		},
	}
	for _, c := range cases {
		actual := a.Get(c.index...)
		if c.expect != actual {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}

}
func TestArrayString(t *testing.T) {
	cases := []struct {
		msg    string
		input  NDArray
		expect string
	}{
		{
			msg: "(2)",
			input: NewNDArray(NewNDShape(2), []float64{
				1, 2,
			}),
			expect: "[1.000000, 2.000000]",
		},
		{
			msg: "(2, 3).Slice(0)",
			input: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Slice(0),
			expect: "[1.000000, 2.000000, 3.000000]",
		},
		{
			msg: "(2, 3).Slice(1)",
			input: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Slice(1),
			expect: "[4.000000, 5.000000, 6.000000]",
		},
		{
			msg: "(2, 3)",
			input: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			expect: "[[1.000000, 2.000000, 3.000000],\n[4.000000, 5.000000, 6.000000]]",
		},
	}

	for _, c := range cases {
		actual := c.input.String()
		if c.expect != actual {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}

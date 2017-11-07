package gocnn

import (
	"reflect"
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
func TestNDArrayString(t *testing.T) {
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
			msg: "(2, 3).Segment(0)",
			input: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Segment(0),
			expect: "[1.000000, 2.000000, 3.000000]",
		},
		{
			msg: "(2, 3).Segment(1)",
			input: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Segment(1),
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
func TestTransposedShape(t *testing.T) {
	cases := []struct {
		msg    string
		shape  NDShape
		index  []int
		expect NDShape
	}{
		{
			msg:    "(2, 3).Traspose(1, 0)",
			shape:  NewNDShape(2, 3),
			index:  []int{1, 0},
			expect: NewNDShape(3, 2),
		},
		{
			msg:    "(2, 3, 4).Traspose(1, 2, 0)",
			shape:  NewNDShape(2, 3, 4),
			index:  []int{1, 2, 0},
			expect: NewNDShape(3, 4, 2),
		},
	}

	for _, c := range cases {
		actual := NewTrShape(c.shape, c.index...)
		if actual.Equals(c.expect) {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}
func TestNDArrayTranspose(t *testing.T) {
	cases := []struct {
		msg    string
		array  NDArray
		index  []int
		expect NDArray
	}{
		{
			msg: "(2, 3).Traspose(1, 0)",
			array: NewNDArray(NewNDShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			index: []int{1, 0},
			expect: NewNDArray(NewNDShape(3, 2), []float64{
				1, 4,
				2, 5,
				3, 6,
			}),
		},
		/*
			{
				msg: "(2, 3, 4).Traspose(1, 2, 0)",
				array: NewNDArray(NewNDShape(2, 3, 4), []float64{
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,

					13, 14, 15, 16,
					17, 18, 19, 20,
					21, 22, 23, 34,
				}),
				index: []int{1, 2, 0},
				expect: NewNDArray(NewNDShape(3, 2, 4), []float64{
					1, 13,
					2, 14,
					3, 15,
					4, 16,

					5, 17,
					6, 18,
					7, 19,
					8, 20,

					9, 21,
					10, 22,
					11, 23,
					12, 24,
				}),
			},
		*/
	}

	for _, c := range cases {
		actual := c.array.Transpose(c.index...)
		if actual.Get(0, 0) != c.array.Get(0, 0) {
			t.Fail()
		}
		if actual.Get(0, 1) != c.array.Get(1, 0) {
			t.Fail()
		}
		if actual.Get(1, 0) != c.array.Get(0, 1) {
			t.Fail()
		}
		if actual.Get(1, 1) != c.array.Get(1, 1) {
			t.Fail()
		}
		if actual.Get(2, 0) != c.array.Get(0, 2) {
			t.Fail()
		}
		if actual.Get(2, 1) != c.array.Get(1, 2) {
			t.Fail()
		}
		{
			act := actual.Shape().AsSlice()
			exp := []int{3, 2}
			if !reflect.DeepEqual(act, exp) {
				t.Fatalf("expect \n%v got \n%v", exp, act)
			}
		}
		{
			act := actual.Segment(0).Get(1)
			exp := 4.0
			if act != exp {
				//t.Fatalf("expect \n%v got \n%v", exp, act)
			}
		}
	}
}

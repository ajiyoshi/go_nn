package nd

import (
	"reflect"
	"testing"
)

func TestArrayGet(t *testing.T) {
	a := NewArray(NewShape(2, 3),
		[]float64{
			1, 2, 3,
			4, 5, 6,
		})

	cases := []struct {
		msg    string
		index  []int
		expect float64
	}{
		{msg: "(0, 0)", index: []int{0, 0}, expect: 1},
		{msg: "(0, 1)", index: []int{0, 1}, expect: 2},
		{msg: "(0, 2)", index: []int{0, 2}, expect: 3},
		{msg: "(1, 0)", index: []int{1, 0}, expect: 4},
		{msg: "(1, 1)", index: []int{1, 1}, expect: 5},
		{msg: "(1, 2)", index: []int{1, 2}, expect: 6},
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
		input  Array
		expect string
	}{
		{
			msg: "(1, 2)",
			input: NewArray(NewShape(2), []float64{
				1, 2,
			}),
			expect: "[1.00, 2.00]",
		},
		{
			msg: "(2, 3).Segment(0)",
			input: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Segment(0),
			expect: "[1.00, 2.00, 3.00]",
		},
		{
			msg: "(2, 3).Segment(1)",
			input: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Segment(1),
			expect: "[4.00, 5.00, 6.00]",
		},
		{
			msg: "(2, 3)",
			input: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			expect: "[[1.00, 2.00, 3.00],\n[4.00, 5.00, 6.00]]",
		},
		{
			msg: "(2, 3).Transpose(1, 0)",
			input: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}).Transpose(1, 0),
			expect: "[[1.00, 4.00],\n[2.00, 5.00],\n[3.00, 6.00]]",
		},
		{
			msg: "(2, 3, 4).Transpose(1, 2, 0).Segment(0)",
			input: NewArray(NewShape(2, 3, 4), []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,

				13, 14, 15, 16,
				17, 18, 19, 20,
				21, 22, 23, 24,
			}).Transpose(2, 0, 1).Segment(0),
			expect: "[[1.00, 5.00, 9.00],\n[13.00, 17.00, 21.00]]",
		},
		{
			msg: "(2, 3, 4).Segment(0).Transpose(1, 0)",
			input: NewArray(NewShape(2, 3, 4), []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,

				13, 14, 15, 16,
				17, 18, 19, 20,
				21, 22, 23, 24,
			}).Segment(0).Transpose(1, 0),
			expect: "[[1.00, 5.00, 9.00],\n[2.00, 6.00, 10.00],\n[3.00, 7.00, 11.00],\n[4.00, 8.00, 12.00]]",
		},
	}

	for _, c := range cases {
		actual := c.input.String()
		if c.expect != actual {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}
func TestArrayTranspose(t *testing.T) {
	cases := []struct {
		msg    string
		array  Array
		index  []int
		expect Array
	}{
		{
			msg: "(2, 3).Traspose(1, 0)",
			array: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			index: []int{1, 0},
			expect: NewArray(NewShape(3, 2), []float64{
				1, 4,
				2, 5,
				3, 6,
			}),
		},
		{
			msg: "(2, 3, 4).Traspose(1, 2, 0)",
			array: NewArray(NewShape(2, 3, 4), []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,

				13, 14, 15, 16,
				17, 18, 19, 20,
				21, 22, 23, 24,
			}),
			index: []int{1, 2, 0},
			expect: NewArray(NewShape(3, 4, 2), []float64{
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
	}

	for _, c := range cases {
		actual := c.array.Transpose(c.index...)
		if !actual.DeepEqual(c.expect) {
			t.Fatalf("expect %v\n%v got %v\n%v", c.expect.Shape(), c.expect, actual.Shape(), actual)
		}
	}
}

func TestArrayIterator(t *testing.T) {
	array := NewArray(NewShape(1, 2, 3), make([]float64, 6))

	cases := []struct {
		expect []int
	}{
		{[]int{0, 0, 0}},
		{[]int{0, 0, 1}},
		{[]int{0, 0, 2}},
		{[]int{0, 1, 0}},
		{[]int{0, 1, 1}},
		{[]int{0, 1, 2}},
	}

	itr := array.Iterator()
	if !itr.OK() {
		t.Fail()
	}
	for _, c := range cases {
		if !itr.OK() {
			t.Fatalf("not ok at %v", c.expect)
		}
		if !reflect.DeepEqual(itr.Index(), c.expect) {
			t.Fatalf("expect %v got %v", c.expect, itr.Index())
		}
		itr.Next()
	}
	if itr.OK() {
		t.Fail()
	}
}

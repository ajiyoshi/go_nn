package main

import (
	"github.com/gonum/matrix/mat64"
	"reflect"
	"testing"
)

func TestTrainBuffer(t *testing.T) {
	buf := NewTrainBuffer(2, 728, 10)
	if buf == nil {
		t.Fail()
	}
	x0 := make([]byte, 728)
	x0[0] = 255
	x1 := make([]byte, 728)
	x1[1] = 255
	buf.LoadX(0, x0)
	buf.LoadX(1, x1)

	buf.LoadT(0, 1)
	buf.LoadT(1, 2)

	mx, mt := buf.Bake()

	X := mat64.NewDense(2, 728, nil)
	X.Set(0, 0, 1.0)
	X.Set(1, 1, 1.0)
	T := mat64.NewDense(2, 10, []float64{
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	})

	if !mat64.EqualApprox(mx, X, 0.01) {
		t.Fail()
	}
	if !mat64.EqualApprox(mt, T, 0.01) {
		t.Fail()
	}

}

func TestSeq(t *testing.T) {
	cases := []struct {
		title string
		x     int

		n int

		expect []int
	}{
		{
			title:  "Seq",
			x:      0,
			n:      5,
			expect: []int{0, 1, 2, 3, 4},
		},
	}
	for _, c := range cases {
		actual := seq(c.x, c.n)
		if !reflect.DeepEqual(actual, c.expect) {
			t.Fatalf("%s expect %v but got %v\n", c.title, c.expect, actual)
		}
	}
}

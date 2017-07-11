package mnist

import (
	"github.com/gonum/matrix/mat64"
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

func TestLabelAsNum(t *testing.T) {
	cases := []struct {
		title  string
		v      []float64
		expect int
	}{
		{
			title:  "label as num",
			v:      []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expect: 0,
		},
		{
			title:  "label as num",
			v:      []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
			expect: 1,
		},
		{
			title:  "label as num",
			v:      []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
			expect: 9,
		},
	}
	for _, c := range cases {
		actual := LabelAsNum(c.v)
		if actual != c.expect {
			t.Fatalf("%s expect %v but got %v\n", c.title, c.expect, actual)
		}
	}
}

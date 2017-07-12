package matrix

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestSubMatrix(t *testing.T) {
	cases := []struct {
		title      string
		m          mat64.Mutable
		i, j, r, c int
		expect     mat64.Mutable
		T          mat64.Matrix
	}{
		{
			title: "sub matrix",
			m: mat64.NewDense(3, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			}),
			i: 0, j: 0, r: 2, c: 3,
			expect: mat64.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			T: mat64.NewDense(3, 2, []float64{
				1, 4,
				2, 5,
				3, 6,
			}),
		},
	}

	for _, c := range cases {
		actual := NewSubMutable(c.m, c.i, c.j, c.r, c.c)

		if !mat64.EqualApprox(c.expect, actual, 0.01) {
			t.Fatalf("%s expect\n%.2g but got\n%.2g\n",
				c.title, mat64.Formatted(c.expect), mat64.Formatted(actual))
		}
		if !mat64.EqualApprox(c.T, actual.T(), 0.01) {
			t.Fatalf("%s expect\n%.2g but got\n%.2g\n",
				c.title, mat64.Formatted(c.T), mat64.Formatted(actual.T()))
		}
	}
}

func TestZeroPadMatrix(t *testing.T) {
	cases := []struct {
		title  string
		m      mat64.Mutable
		n      int
		expect mat64.Mutable
		T      mat64.Matrix
	}{
		{
			title: "sub matrix",
			m: mat64.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			n: 1,
			expect: mat64.NewDense(4, 5, []float64{
				0, 0, 0, 0, 0,
				0, 1, 2, 3, 0,
				0, 4, 5, 6, 0,
				0, 0, 0, 0, 0,
			}),
			T: mat64.NewDense(5, 4, []float64{
				0, 0, 0, 0,
				0, 1, 4, 0,
				0, 2, 5, 0,
				0, 3, 6, 0,
				0, 0, 0, 0,
			}),
		},
	}

	for _, c := range cases {
		actual := NewZeroPadMutable(c.m, c.n)

		if !mat64.EqualApprox(c.expect, actual, 0.01) {
			t.Fatalf("%s expect %.2g but got %.2g\n",
				c.title, mat64.Formatted(c.expect), mat64.Formatted(actual))
		}

		if !mat64.EqualApprox(c.T, actual.T(), 0.01) {
			t.Fatalf("%s expect %.2g but got %.2g\n",
				c.title, mat64.Formatted(c.T), mat64.Formatted(actual.T()))
		}
	}

}

func TestTransposeMutable(t *testing.T) {
	cases := []struct {
		title  string
		m      mat64.Mutable
		expect mat64.Mutable
		T      mat64.Mutable
	}{
		{
			title: "sub matrix",
			m: mat64.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
			expect: mat64.NewDense(3, 2, []float64{
				1, 4,
				2, 5,
				3, 6,
			}),
			T: mat64.NewDense(2, 3, []float64{
				1, 2, 3,
				4, 5, 6,
			}),
		},
	}

	for _, c := range cases {
		actual := NewTransposeMutable(c.m)

		if !mat64.EqualApprox(c.expect, actual, 0.01) {
			t.Fatalf("%s expect %.2g but got %.2g\n",
				c.title, mat64.Formatted(c.expect), mat64.Formatted(actual))
		}

		if !mat64.EqualApprox(c.T, actual.T(), 0.01) {
			t.Fatalf("%s expect %.2g but got %.2g\n",
				c.title, mat64.Formatted(c.T), mat64.Formatted(actual.T()))
		}
	}

}

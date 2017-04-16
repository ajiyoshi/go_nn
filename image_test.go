package main

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestIm2col(t *testing.T) {
	cases := []struct {
		title  string
		img    *SimpleStrage
		expect mat64.Matrix
		pad    int
		stride int
		fr     int
		fc     int
	}{
		{
			title: "im2col",
			img: NewImages(1, 2, 3, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				10, 20, 30,
				40, 50, 60,
				70, 80, 90,
			}),
			expect: mat64.NewDense(4, 8, []float64{
				1, 2, 4, 5, 10, 20, 40, 50,
				2, 3, 5, 6, 20, 30, 50, 60,
				4, 5, 7, 8, 40, 50, 70, 80,
				5, 6, 8, 9, 50, 60, 80, 90,
			}),
			fr: 2, fc: 2,
			stride: 1,
			pad:    0,
		},
		{
			title: "im2col",
			img: NewImages(1, 2, 2, 2, []float64{
				1, 2,
				3, 4,

				5, 6,
				7, 8,
			}),
			expect: mat64.NewDense(9, 8, []float64{
				0, 0, 0, 1, 0, 0, 0, 5,
				0, 0, 1, 2, 0, 0, 5, 6,
				0, 0, 2, 0, 0, 0, 6, 0,
				0, 1, 0, 3, 0, 5, 0, 7,
				1, 2, 3, 4, 5, 6, 7, 8,
				2, 0, 4, 0, 6, 0, 8, 0,
				0, 3, 0, 0, 0, 7, 0, 0,
				3, 4, 0, 0, 7, 8, 0, 0,
				4, 0, 0, 0, 8, 0, 0, 0,
			}),
			fr: 2, fc: 2,
			stride: 1,
			pad:    1,
		},
		{
			title: "im2col",
			img: NewImages(2, 2, 3, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				10, 20, 30,
				40, 50, 60,
				70, 80, 90,

				70, 80, 90,
				40, 50, 60,
				10, 20, 30,

				7, 8, 9,
				4, 5, 6,
				1, 2, 3,
			}),
			expect: mat64.NewDense(8, 8, []float64{
				0, 0, 0, 1, 0, 0, 0, 10,
				0, 0, 2, 3, 0, 0, 20, 30,
				0, 4, 0, 7, 0, 40, 0, 70,
				5, 6, 8, 9, 50, 60, 80, 90,
				0, 0, 0, 70, 0, 0, 0, 7,
				0, 0, 80, 90, 0, 0, 8, 9,
				0, 40, 0, 10, 0, 4, 0, 1,
				50, 60, 20, 30, 5, 6, 2, 3,
			}),
			fr: 2, fc: 2,
			stride: 2,
			pad:    1,
		},
	}

	for _, c := range cases {
		actual := Im2col(c.img, c.fr, c.fc, c.stride, c.pad)

		if !mat64.EqualApprox(c.expect, actual, 0.01) {
			t.Fatalf("%s expect \n%.2g but got \n%.2g\n",
				c.title, mat64.Formatted(c.expect), mat64.Formatted(actual))
		}
	}
}

func TestSubMatrix(t *testing.T) {
	cases := []struct {
		title      string
		m          mat64.Matrix
		i, j, r, c int
		expect     mat64.Matrix
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
		actual := NewSubMatrix(c.m, c.i, c.j, c.r, c.c)

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

func TestZeroPadMatrix(t *testing.T) {
	cases := []struct {
		title  string
		m      mat64.Matrix
		n      int
		expect mat64.Matrix
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
		actual := NewZeroPadMatrix(c.m, c.n)

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

package main

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestSumCols(t *testing.T) {
	cases := []struct {
		title string
		input mat64.Matrix
		cols  *mat64.Vector
		rows  *mat64.Vector
	}{
		{
			title: "TestSumCols",
			input: mat64.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
			cols: mat64.NewVector(2, []float64{9, 12}),
			rows: mat64.NewVector(3, []float64{3, 7, 11}),
		},
	}

	for _, c := range cases {
		cols := SumCols(c.input, nil)
		if !mat64.EqualApprox(cols, c.cols, 0.001) {
			t.Fatalf("%s expect %v but got %v", c.title, c.cols, cols)
		}

		rows := SumRows(c.input, nil)
		if !mat64.EqualApprox(rows, c.rows, 0.001) {
			t.Fatalf("%s expect %v but got %v", c.title, c.rows, rows)
		}
	}
}

func TestNormalizeEachRow(t *testing.T) {
	cases := []struct {
		title  string
		input  *mat64.Dense
		expect mat64.Matrix
	}{
		{
			title: "TestNormalizeEachRowe",
			input: mat64.NewDense(3, 2, []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
			expect: mat64.NewDense(3, 2, []float64{
				1.0 / 3, 2.0 / 3,
				3.0 / 7, 4.0 / 7,
				5.0 / 11, 6.0 / 11,
			}),
		},
	}

	for _, c := range cases {
		NormalizeEachRow(c.input)
		if !mat64.EqualApprox(c.input, c.expect, 0.001) {
			t.Fatalf("%s expect %v but got %v", c.title, c.expect, c.input)
		}
	}
}

func TestSoftMaxV(t *testing.T) {
	for _, c := range []struct {
		title  string
		input  *mat64.Vector
		expect *mat64.Vector
	}{
		{
			title:  "TestSoftMaxV",
			input:  mat64.NewVector(3, []float64{1, 2, 3}),
			expect: mat64.NewVector(3, []float64{0.0900, 0.2447, 0.6652}),
		},
	} {
		actual := SoftMaxV(c.input)
		if !mat64.EqualApprox(actual, c.expect, 0.0001) {
			t.Fatalf("%s expect %v but got %v", c.title, c.expect, actual)
		}
	}
}

func TestSoftMax(t *testing.T) {
	for _, c := range []struct {
		title  string
		input  mat64.Matrix
		expect mat64.Matrix
	}{
		{
			title:  "TestSoftMax",
			input:  mat64.NewDense(1, 3, []float64{1, 2, 3}),
			expect: mat64.NewDense(1, 3, []float64{0.0900, 0.2447, 0.6652}),
		},
		{
			title: "TestSoftMax",
			input: mat64.NewDense(2, 3, []float64{
				1, 2, 3,
				-1, 0, 1000,
			}),
			expect: mat64.NewDense(2, 3, []float64{
				0.0900, 0.2447, 0.6652,
				0, 0, 1,
			}),
		},
	} {
		actual := SoftMax(c.input)
		if !mat64.EqualApprox(actual, c.expect, 0.0001) {
			t.Fatalf("%s expect %v but got %v", c.title, c.expect, actual)
		}
	}
}

func TestCrossEntrpyError(t *testing.T) {
	for _, c := range []struct {
		title  string
		y      mat64.Matrix
		t      mat64.Matrix
		expect float64
	}{
		{
			title:  "hoge",
			y:      mat64.NewDense(1, 10, []float64{0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0}),
			t:      mat64.NewDense(1, 10, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}),
			expect: 0.51082545709933803,
		},
	} {
		actual := CrossEntropyError(c.y, c.t)
		if !NealyEqual(actual, c.expect) {
			t.Fatalf("%s expect %v but got %v (%v)", c.title, c.expect, actual, ErrorRate(c.expect, actual))
		}
	}
}

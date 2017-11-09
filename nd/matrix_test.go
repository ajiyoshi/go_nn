package nd

import (
	mat "github.com/gonum/matrix/mat64"
	"testing"
)

func TestArrayMatrix(t *testing.T) {
	cases := []struct {
		msg      string
		generate func() (actual, expect mat.Matrix)
	}{
		{
			msg: "(2) -> (1, 2)",
			generate: func() (mat.Matrix, mat.Matrix) {
				row, col := 1, 2

				expect := NewArray(NewShape(2), []float64{
					4, 5,
				}).AsMatrix(row, col)
				actual := mat.NewDense(row, col, []float64{
					4, 5,
				})
				return expect, actual
			},
		},
		{
			msg: "(2) -> (2, 1)",
			generate: func() (mat.Matrix, mat.Matrix) {
				row, col := 2, 1

				expect := NewArray(NewShape(2), []float64{
					4, 5,
				}).AsMatrix(row, col)
				actual := mat.NewDense(row, col, []float64{
					4,
					5,
				})
				return expect, actual
			},
		},
		{
			msg: "(2, 2, 2) -> (2, 4)",
			generate: func() (mat.Matrix, mat.Matrix) {
				row, col := 2, 4

				expect := NewArray(NewShape(2, 2, 2), []float64{
					1, 2,
					3, 4,

					5, 6,
					7, 8,
				}).AsMatrix(row, col)
				actual := mat.NewDense(row, col, []float64{
					1, 2, 3, 4,
					5, 6, 7, 8,
				})
				return expect, actual
			},
		},
	}

	for _, c := range cases {
		expect, actual := c.generate()
		if !mat.EqualApprox(expect, actual, 0.001) {
			t.Fatalf("(%s) expect \n%v but actual \n%v", c.msg, mat.Formatted(expect), mat.Formatted(actual))
		}
	}
}

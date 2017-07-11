package optimizer

import (
	"github.com/gonum/matrix/mat64"
	"testing"
)

func TestMomemtum(t *testing.T) {
	m := NewMomentum(1, 1)

	param := mat64.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	grad := mat64.NewDense(2, 3, []float64{
		3, 2, 1,
		6, 5, 4,
	})

	m.UpdateWeight(param, grad)
}

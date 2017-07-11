package gocnn

import (
	"github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn"
)

type NeuralNetLayers interface {
	Layers() []Layer
	Last() LastLayer
}

type NeuralNet struct {
	layers NeuralNetLayers
}

func NewNeuralNet(ls NeuralNetLayers) *NeuralNet {
	return &NeuralNet{ls}
}

func (nn *NeuralNet) Layers() []Layer {
	return nn.layers.Layers()
}

func (nn *NeuralNet) Predict(x mat64.Matrix) mat64.Matrix {
	for _, layer := range nn.Layers() {
		x = layer.Forward(x)
	}
	return x
}

func (nn *NeuralNet) Loss(x, t mat64.Matrix) float64 {
	y := nn.Predict(x)
	return nn.layers.Last().Forward(y, t)
}

func (nn *NeuralNet) Accracy(x, t mat64.Matrix) float64 {
	y := nn.Predict(x)
	r, _ := x.Dims()
	ok := 0.0
	for i := 0; i < r; i++ {
		a := gocnn.Argmax(mat64.Row(nil, i, y))
		b := gocnn.Argmax(mat64.Row(nil, i, t))
		if a == b {
			ok++
		}
	}
	return ok / float64(r)
}

func (nn *NeuralNet) BackProp() mat64.Matrix {
	dout := nn.layers.Last().Backward(1)
	for _, layer := range LayerReverse(nn.Layers()) {
		dout = layer.Backward(dout)
	}
	return dout
}

func (nn *NeuralNet) Update() {
	for _, layer := range nn.Layers() {
		layer.Update()
	}
}
func (nn *NeuralNet) Train(x, t mat64.Matrix) float64 {
	loss := nn.Loss(x, t)
	nn.BackProp()
	nn.Update()
	return loss
}

func LayerReverse(ls []Layer) []Layer {
	len := len(ls)
	ret := make([]Layer, len)
	for i, x := range ls {
		ret[len-i-1] = x
	}
	return ret
}

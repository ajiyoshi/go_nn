package single

import (
	"github.com/gonum/matrix/mat64"
)

type NNImpl interface {
	Layers() []Layer
	Last() LastLayer
}

var _ NNImpl = &TwoLayerNN{}

type NeuralNet struct {
	impl NNImpl
}

func NewNeuralNet(imp NNImpl) *NeuralNet {
	return &NeuralNet{imp}
}

func (nn *NeuralNet) Layers() []Layer {
	return nn.impl.Layers()
}
func (nn *NeuralNet) Predict(x *mat64.Vector) *mat64.Vector {
	for _, layer := range nn.Layers() {
		x = layer.Forward(x)
	}
	return x
}
func (nn *NeuralNet) Loss(x, t *mat64.Vector) float64 {
	y := nn.Predict(x)
	return nn.impl.Last().Forward(y, t)
}
func (nn *NeuralNet) BackProp() *mat64.Vector {
	dout := nn.impl.Last().Backward(1)
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
func (nn *NeuralNet) Train(x, t *mat64.Vector) float64 {
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

package main

import (
	"github.com/gonum/matrix/mat64"
)

type BatchNeuralNetLayers interface {
	Layers() []BatchLayer
	Last() BatchLastLayer
}

type BatchNeuralNet struct {
	layers BatchNeuralNetLayers
}

func (nn *BatchNeuralNet) Layers() []BatchLayer {
	return nn.layers.Layers()
}

func (nn *BatchNeuralNet) Predict(x mat64.Matrix) mat64.Matrix {
	for _, layer := range nn.Layers() {
		x = layer.Forward(x)
	}
	return x
}

func (nn *BatchNeuralNet) Loss(x, t mat64.Matrix) float64 {
	y := nn.Predict(x)
	return nn.layers.Last().Forward(y, t)
}

func (nn *BatchNeuralNet) BackProp() mat64.Matrix {
	dout := nn.layers.Last().Backward(1)
	for _, layer := range LayerReverseB(nn.Layers()) {
		dout = layer.Backward(dout)
	}
	return dout
}

func (nn *BatchNeuralNet) Update() {
	for _, layer := range nn.Layers() {
		layer.Update()
	}
}
func (nn *BatchNeuralNet) Train(x, t mat64.Matrix) float64 {
	loss := nn.Loss(x, t)
	nn.BackProp()
	nn.Update()
	return loss
}

func LayerReverseB(ls []BatchLayer) []BatchLayer {
	len := len(ls)
	ret := make([]BatchLayer, len)
	for i, x := range ls {
		ret[len-i-1] = x
	}
	return ret
}

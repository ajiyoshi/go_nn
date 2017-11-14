package gocnn

import (
	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/batch"
)

type SimpleCNN struct {
	imageLayers   []ImageLayer
	imageToMatrix ImageToMatrix
	nn            *batch.NeuralNet
}

func (cnn *SimpleCNN) Forward(img Image) mat.Matrix {
	for _, layer := range cnn.imageLayers {
		img = layer.Forward(img)
	}
	return cnn.imageToMatrix.Forward(img)
}
func (cnn *SimpleCNN) Backword(m mat.Matrix) Image {
	dout := cnn.imageToMatrix.Backword(m)
	for _, layer := range LayerReverse(cnn.imageLayers) {
		dout = layer.Backword(dout)
	}
	return dout
}

func (cnn *SimpleCNN) Predict(img Image) mat.Matrix {
	m := cnn.Forward(img)
	return cnn.nn.Predict(m)
}
func (cnn *SimpleCNN) Loss(img Image, t mat.Matrix) float64 {
	x := cnn.Forward(img)
	return cnn.nn.Loss(x, t)
}
func (cnn *SimpleCNN) BackProp() Image {
	dout := cnn.nn.BackProp()
	return cnn.Backword(dout)
}
func (cnn *SimpleCNN) Accracy(img Image, t mat.Matrix) float64 {
	x := cnn.Forward(img)
	return cnn.nn.Accracy(x, t)
}
func (cnn *SimpleCNN) Update() {
	for _, layer := range cnn.imageLayers {
		layer.Update()
	}
	cnn.nn.Update()
}
func (cnn *SimpleCNN) Train(img Image, t mat.Matrix) float64 {
	loss := cnn.Loss(img, t)
	cnn.BackProp()
	cnn.Update()
	return loss
}

type ImageLayer interface {
	Forward(Image) Image
	Backword(Image) Image
	Update()
	Equals(ImageLayer) bool
}

var (
	_ ImageLayer = (*Convolution)(nil)
	_ ImageLayer = (*Pooling)(nil)
	_ ImageLayer = (*ReLU)(nil)
)

type ImageToMatrix struct {
	shape *Shape
}

func (c *ImageToMatrix) Forward(x Image) mat.Matrix {
	c.shape = x.Shape()
	return x.ToMatrix(c.shape.N, c.shape.Size()/c.shape.N)
}
func (c *ImageToMatrix) Backword(dout mat.Matrix) Image {
	return NewReshaped(c.shape, dout)
}

func LayerReverse(ls []ImageLayer) []ImageLayer {
	len := len(ls)
	ret := make([]ImageLayer, len)
	for i, x := range ls {
		ret[len-i-1] = x
	}
	return ret
}

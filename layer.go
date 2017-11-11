package gocnn

import (
	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/optimizer"
)

type Convolution struct {
	Weight    Image
	Bias      *mat.Vector
	Stride    int
	Pad       int
	Optimizer optimizer.Optimizer

	dWeight Image
	dBias   *mat.Vector
	col     mat.Matrix
	colW    mat.Matrix
	x       Image
}

func NewConvolution(s *Shape, stride, pad int, opt optimizer.Optimizer) *Convolution {
	return &Convolution{
		Weight:    NewRandomImage(s, WeightInitStd),
		Bias:      mat.NewVector(s.N, nil),
		Stride:    stride,
		Pad:       pad,
		Optimizer: opt,
	}
}
func (c *Convolution) Forward(x Image) Image {
	xs := x.Shape()
	ws := c.Weight.Shape()

	if xs.Ch != ws.Ch {
		//panic("number of channels was not match")
	}

	outRow := 1 + (xs.Row+2*c.Pad-ws.Row)/c.Stride
	outCol := 1 + (xs.Row+2*c.Pad-ws.Row)/c.Stride

	// col : (xs.n*outRow*outCol, xs.ch*ws.row*ws.col)
	c.col = Im2col(x, ws.Row, ws.Col, c.Stride, c.Pad)
	// colW : (ws.ch*ws.row*ws.col, ws.n)
	c.colW = c.Weight.Matrix().T()
	c.x = x

	// ret : (xs.n*outRow*outCol, ws.n)
	ret := mul(c.col, c.colW)
	ret.Apply(func(i, j int, val float64) float64 {
		return c.Bias.At(j, 0) + val
	}, ret)

	return NewReshaped(NewShape(xs.N, outRow, outCol, ws.N), ret).Transpose(0, 3, 1, 2)
}

func (c *Convolution) Backword(doutImg Image) Image {
	/*
		FN, C, FH, FW = self.W.shape
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)

		self.db = np.sum(dout, axis=0)
		self.dW = np.dot(self.col.T, dout)
		self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

		dcol = np.dot(dout, self.col_W.T)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

		return dx
	*/
	s := c.Weight.Shape()
	dout := doutImg.Transpose(0, 2, 3, 1).ToMatrix(doutImg.Size()/s.N, s.N)

	dWeight := mul(c.col.T(), dout)
	c.dWeight = NewReshaped(s, dWeight.T())
	c.dBias = matrix.SumCols(dout, c.dBias)

	dcol := mul(dout, c.colW.T())
	dx := Col2im(dcol, c.x.Shape(), s.Row, s.Col, c.Stride, c.Pad)

	return dx
}

func (c *Convolution) Update() {
	c.Optimizer.UpdateWeightArray(c.Weight.ToArray(), c.dWeight.ToArray())
	c.Optimizer.UpdateBias(c.Bias, c.dBias)
}

type Pooling struct {
	Row    int
	Col    int
	Stride int
	Pad    int

	argmax []int
	x      Image
}

func (p *Pooling) Forward(x Image) Image {
	/*
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)

		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)

		arg_max = np.argmax(col, axis=1)
		out = np.max(col, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

		self.x = x
		self.arg_max = arg_max
	*/
	tmp := Im2col(x, p.Row, p.Col, p.Stride, p.Pad)
	col := ReshapeMatrix(-1, p.Row*p.Col, tmp)

	p.argmax = argmaxEachRow(col)
	p.x = x

	out := maxEachRow(col)

	s := x.Shape()
	outRow := 1 + (s.Row-p.Row)/p.Stride
	outCol := 1 + (s.Col-p.Col)/p.Stride
	return NewImages(NewShape(s.N, outRow, outCol, s.Ch), out).Transpose(0, 3, 1, 2)
}

func (p *Pooling) Backword(doutImage Image) Image {
	/*
		dout = dout.transpose(0, 2, 3, 1)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size,))

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

		return dx
	*/
	dout := doutImage.Transpose(0, 2, 3, 1)
	poolSize := p.Row * p.Col
	buf := make([]float64, dout.Size()*poolSize)
	tmp := mat.NewDense(dout.Size(), poolSize, buf)
	flat := mat.Row(nil, 0, dout.ToMatrix(1, dout.Size()))
	for i, x := range flat {
		j := p.argmax[i]
		tmp.Set(i, j, x)
	}
	s := dout.Shape()
	dmax := mat.NewDense(s.N*s.Ch*s.Row, s.Col*poolSize, buf)

	return Col2im(dmax, p.x.Shape(), p.Row, p.Col, p.Stride, p.Pad)
}

func (p *Pooling) Update() {}

type ReLU struct {
	mask *mat.Dense
}

func (r *ReLU) Forward(x Image) Image {
	s := x.Shape()
	r.mask = mat.NewDense(s.N, s.Size()/s.N, nil)
	m := x.ToMatrix(s.N, s.Size()/s.N)

	r.mask.Apply(func(i, j int, v float64) float64 {
		if v < 0 {
			return 0
		} else {
			return 1
		}
	}, m)

	r.mask.MulElem(r.mask, m)

	return NewReshaped(s, r.mask)
}
func (r *ReLU) Backword(dout Image) Image {
	row, col := r.mask.Dims()
	m := dout.ToMatrix(row, col)
	r.mask.MulElem(r.mask, m)
	return NewReshaped(dout.Shape(), r.mask)
}

func (r *ReLU) Update() {}

func mul(x, y mat.Matrix) *mat.Dense {
	var ret mat.Dense
	ret.Mul(x, y)
	return &ret
}

func argmaxEachRow(m mat.Matrix) []int {
	row, col := m.Dims()
	buf := make([]float64, col)
	ret := make([]int, row)
	for i := 0; i < row; i++ {
		mat.Row(buf, i, m)
		ret[i] = matrix.Argmax(buf)
	}
	return ret
}
func maxEachRow(m mat.Matrix) []float64 {
	row, col := m.Dims()
	buf := make([]float64, col)
	ret := make([]float64, row)
	for i := 0; i < row; i++ {
		mat.Row(buf, i, m)
		ret[i] = mat.Max(mat.NewVector(col, buf))
	}
	return ret
}

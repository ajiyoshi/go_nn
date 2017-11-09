package gocnn

import (
	mat "github.com/gonum/matrix/mat64"

	"github.com/ajiyoshi/gocnn/matrix"
	"github.com/ajiyoshi/gocnn/optimizer"
)

type Convolution struct {
	Weight ImageStrage
	Bias   *mat.Vector
	Stride int
	Pad    int

	dWeight   ImageStrage
	dBias     *mat.Vector
	col       mat.Matrix
	colW      mat.Matrix
	x         ImageStrage
	optimizer optimizer.Optimizer
}

func (c *Convolution) Forward(x ImageStrage) ImageStrage {
	xs := x.Shape()
	ws := c.Weight.Shape()

	if xs.ch != ws.ch {
		panic("number of channels was not match")
	}

	outRow := int(1 + (xs.row+2*c.Pad-ws.row)/c.Stride)
	outCol := int(1 + (xs.row+2*c.Pad-ws.row)/c.Stride)

	// col : (xs.n*outRow*outCol, xs.ch*ws.row*ws.col)
	c.col = Im2col(x, ws.row, ws.col, c.Stride, c.Pad)
	// colW : (ws.ch*ws.row*ws.col, ws.n)
	c.colW = c.Weight.Matrix().T()
	c.x = x

	// ret : (xs.n*outRow*outCol, ws.n)
	ret := mul(c.col, c.colW)
	ret.Apply(func(i, j int, val float64) float64 {
		return c.Bias.At(j, 0) + val
	}, ret)

	return NewReshaped([]int{xs.n, outRow, outCol, ws.n}, ret).Transpose(0, 3, 1, 2)
}

func (c *Convolution) Backword(doutImg ImageStrage) ImageStrage {
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
	dout := doutImg.Transpose(0, 2, 3, 1).ToMatrix(doutImg.Size()/s.n, s.n)

	c.dBias = matrix.SumCols(dout, c.dBias)
	dWeight := mul(c.col.T(), dout)
	c.dWeight = NewReshaped([]int{s.n, s.ch, s.row, s.col}, dWeight.T())

	dcol := mul(dout, c.colW.T())
	dx := Col2im(dcol, c.x.Shape(), s.row, s.col, c.Stride, c.Pad)

	return dx
}

type Pooling struct {
	Row    int
	Col    int
	Stride int
	Pad    int

	argmax []int
	x      ImageStrage
}

func (p *Pooling) Forwad(x ImageStrage) ImageStrage {
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
	outRow := 1 + (s.row-p.Row)/p.Stride
	outCol := 1 + (s.col-p.Col)/p.Stride
	return NewReshaped([]int{s.n, outRow, outCol, s.ch}, out).Transpose(0, 3, 1, 2)
}

func (p *Pooling) Backword(doutImage ImageStrage) ImageStrage {
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
	dmax := mat.NewDense(dout.Size(), poolSize, nil)
	flat := mat.Row(nil, 0, dout.ToMatrix(1, dout.Size()))
	for i, x := range flat {
		j := p.argmax[i]
		dmax.Set(i, j, x)
	}

	return Col2im(dmax, p.x.Shape(), p.Row, p.Col, p.Stride, p.Pad)
}

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
func maxEachRow(m mat.Matrix) mat.Matrix {
	row, col := m.Dims()
	buf := make([]float64, col)
	ret := make([]float64, row)
	for i := 0; i < row; i++ {
		mat.Row(buf, i, m)
		ret[i] = mat.Max(mat.NewVector(col, buf))
	}
	return mat.NewDense(row, 1, ret)
}

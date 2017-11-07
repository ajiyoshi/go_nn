package gocnn

type NormalNDShape struct {
	ds []int
}
type NDShape interface {
	Index(is ...int) int
	AsSlice() []int
}
type NDArray interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() NDShape
}

var (
	_ NDArray = (*NormalND)(nil)
	_ NDShape = (*NormalNDShape)(nil)
)

type NormalND struct {
	data  []float64
	shape NDShape
}

func NewNormalND(s *NormalNDShape, data []float64) *NormalND {
	return &NormalND{
		shape: s,
		data:  data,
	}
}

func NewShapeND(ds ...int) *NormalNDShape {
	return &NormalNDShape{
		ds: ds,
	}
}

func (s *NormalNDShape) AsSlice() []int {
	return s.ds
}
func (s *NormalNDShape) Index(is ...int) int {
	ret := 0
	ds := s.Coefficient()
	for i, x := range is {
		ret += ds[i] * x
	}
	return ret
}
func (s *NormalNDShape) Coefficient() []int {
	/*
		[]int{ (d1*d2*...*dn), (d2*...*dn), ... dn, 1 }
	*/
	buf := s.ds
	length := len(buf)
	ret := make([]int, length)
	acc := 1
	for i := 0; i < length; i++ {
		ret[length-i-1] = acc
		//末尾を掛ける
		acc *= buf[len(buf)-1]
		//末尾を削除
		buf = buf[:len(buf)-1]
	}
	return ret
}

func (x *NormalND) Get(is ...int) float64 {
	i := x.shape.Index(is...)
	return x.data[i]
}

func (x *NormalND) Set(v float64, is ...int) {
	i := x.shape.Index(is...)
	x.data[i] = v
}
func (x *NormalND) At(i int) NDArray {
	return nil
}

func (x *NormalND) Shape() NDShape {
	return x.shape
}

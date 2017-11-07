package gocnn

type NDArray interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() NDShape
}
type NDShape interface {
	Index(is ...int) int
	AsSlice() []int
}

var (
	_ NDArray = (*ndArray)(nil)
	_ NDShape = (*ndShape)(nil)
)

type ndArray struct {
	data  []float64
	shape NDShape
}
type ndShape struct {
	ds []int
}

func NewNDArray(s NDShape, data []float64) *ndArray {
	return &ndArray{
		shape: s,
		data:  data,
	}
}
func (x *ndArray) Get(is ...int) float64 {
	i := x.shape.Index(is...)
	return x.data[i]
}

func (x *ndArray) Set(v float64, is ...int) {
	i := x.shape.Index(is...)
	x.data[i] = v
}
func (x *ndArray) At(i int) NDArray {
	return nil
}

func (x *ndArray) Shape() NDShape {
	return x.shape
}

func NewNDShape(ds ...int) *ndShape {
	return &ndShape{
		ds: ds,
	}
}
func (s *ndShape) AsSlice() []int {
	return s.ds
}
func (s *ndShape) Index(is ...int) int {
	ret := 0
	ds := s.Coefficient()
	for i, x := range is {
		ret += ds[i] * x
	}
	return ret
}
func (s *ndShape) Coefficient() []int {
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

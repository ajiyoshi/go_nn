package gocnn

type ShapeND struct {
	ds []int
}
type ArrayND interface {
	Get(is ...int) float64
	Set(x float64, is ...int)
	Shape() *ShapeND
}

var (
	_ ArrayND = (*NormalND)(nil)
)

type NormalND struct {
	data  []float64
	index IndexerND
}

type IndexerND interface {
	At(is ...int) int
	Shape() *ShapeND
}

type NormalIndexerND struct {
	shape *ShapeND
}

func NewNormalIndexerND(s *ShapeND) *NormalIndexerND {
	return &NormalIndexerND{s}
}
func (i *NormalIndexerND) At(is ...int) int {
	s := i.Shape()
	return s.Index(is...)
}
func (i NormalIndexerND) Shape() *ShapeND {
	return i.shape
}

func NewNormalND(s *ShapeND, data []float64) *NormalND {
	return &NormalND{
		index: NewNormalIndexerND(s),
		data:  data,
	}
}

func NewShapeND(ds ...int) *ShapeND {
	return &ShapeND{
		ds: ds,
	}
}

func (s *ShapeND) Index(is ...int) int {
	ret := 0
	ds := s.Coefficient()
	for i, x := range is {
		ret += ds[i] * x
	}
	return ret
}
func (s *ShapeND) Coefficient() []int {
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
	i := x.index.At(is...)
	return x.data[i]
}

func (x *NormalND) Set(v float64, is ...int) {
	i := x.index.At(is...)
	x.data[i] = v
}

func (x *NormalND) Shape() *ShapeND {
	return x.index.Shape()
}

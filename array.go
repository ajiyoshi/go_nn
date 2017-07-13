package gocnn

type Shape4D struct {
	d0, d1, d2, d3 int
}
type Array4D interface {
	Get(i0, i1, i2, i3 int) float64
	Set(i0, i1, i2, i3 int, x float64)
	Shape() Shape4D
}

var (
	_ Array4D = (*Normal4D)(nil)
)

type Normal4D struct {
	data  []float64
	index Indexer
}

type Indexer interface {
	At(i0, i1, i2, i3 int) int
	Shape() Shape4D
}

type NormalIndexer struct {
	shape Shape4D
}

func NewNormalIndexer(s Shape4D) *NormalIndexer {
	return &NormalIndexer{s}
}
func (i *NormalIndexer) At(i0, i1, i2, i3 int) int {
	s := i.Shape()
	return i0*(s.d1*s.d2*s.d3) + i1*(s.d2*s.d3) + i2*s.d3 + i3
}
func (i NormalIndexer) Shape() Shape4D {
	return i.shape
}

func NewNormal4D(s Shape4D, data []float64) *Normal4D {
	return &Normal4D{
		index: NewNormalIndexer(s),
		data:  data,
	}
}

func Index(i0, i1, i2, i3 int, s Shape4D) int {
	return i0*(s.d1*s.d2*s.d3) + i1*(s.d2*s.d3) + i2*s.d3 + i3
}

func (x *Normal4D) Get(i0, i1, i2, i3 int) float64 {
	i := x.index.At(i0, i1, i2, i3)
	return x.data[i]
}

func (x *Normal4D) Set(i0, i1, i2, i3 int, v float64) {
	i := x.index.At(i0, i1, i2, i3)
	x.data[i] = v
}

func (x *Normal4D) Shape() Shape4D {
	return x.index.Shape()
}

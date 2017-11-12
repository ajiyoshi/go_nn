package nd

func (x Shape) Iterator() Iterator {
	coef := Coefficient(x)
	return &ndArrayIterator{
		i:     0,
		shape: x,
		coef:  coef,
		max:   x.Size(),
		buf:   make([]int, len(x)),
	}
}

type ndArrayIterator struct {
	i     int
	shape Shape
	max   int
	coef  []int
	buf   []int
}

func (itr *ndArrayIterator) OK() bool {
	return itr.i < itr.max
}
func (itr *ndArrayIterator) Next() {
	itr.i++

	for i := len(itr.buf) - 1; i >= 0; i-- {
		if itr.buf[i]+1 < itr.shape[i] {
			itr.buf[i]++
			clear(itr.buf[i+1:])
			return
		}
	}
}
func (itr *ndArrayIterator) Reset() {
	itr.i = 0
	clear(itr.buf)
}
func (itr *ndArrayIterator) Index() []int {
	return itr.buf
}
func clear(idx []int) {
	for i, _ := range idx {
		idx[i] = 0
	}
}

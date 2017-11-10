package nd

func (x Shape) Iterator() Iterator {
	coef := Coefficient(x)
	max := coef[0]
	return &ndArrayIterator{
		i:     0,
		shape: x,
		coef:  coef,
		max:   max,
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
	for i, c := range itr.coef {
		itr.buf[i] = itr.i / c % itr.shape[i]
	}
}
func (itr *ndArrayIterator) Index() []int {
	return itr.buf
}

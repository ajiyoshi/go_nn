package nd

type NormalIndexer struct {
	shape Shape
}
type TransposeIndexer struct {
	table  []int
	origin Indexer
}
type SubIndexer struct {
	fixed  int
	origin Indexer
}

func (x *NormalIndexer) At(is ...int) int {
	ret := 0
	coef := Coefficient(x.shape)
	for i, x := range is {
		ret += coef[i] * x
	}
	return ret
}

func (x *TransposeIndexer) At(is ...int) int {
	return x.origin.At(indexConvert(is, x.table)...)
}
func indexConvert(origin, table []int) []int {
	/*
		unspeakable spec
			index(1, 2) (1, 0) -> index(2, 1)
			index(0, 0, 1) (1, 2, 0) -> index(1, 0, 0)
	*/
	ret := make([]int, len(origin))
	for i, at := range table {
		ret[at] = origin[i]
	}
	return ret
}

func (x *SubIndexer) At(is ...int) int {
	return x.origin.At(Cons(x.fixed, is)...)
}

func Cons(car int, cdr []int) []int {
	ret := make([]int, len(cdr)+1)
	ret[0], ret = car, append(ret[:1], cdr...)
	return ret
}

func Coefficient(dims []int) []int {
	/*
		[]int{ d0, d1, d2, ... dn }
		-> []int{ (d1*d2*...*dn), (d2*...*dn), ... dn, 1 }
	*/
	ptr, length := dims, len(dims)
	ret := make([]int, length)
	acc := 1
	for i := 0; i < length; i++ {
		ret[length-i-1] = acc
		//末尾を掛ける
		acc *= ptr[len(ptr)-1]
		//末尾を削除
		ptr = ptr[:len(ptr)-1]
	}
	return ret
}

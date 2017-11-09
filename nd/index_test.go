package nd

import (
	"reflect"
	"testing"
)

func TestIndexConvert(t *testing.T) {

	cases := []struct {
		msg    string
		index  []int
		table  []int
		expect []int
	}{
		{msg: "index{4, 5, 6} by{0, 1, 2} -> index{4, 5, 6}",
			index: []int{4, 5, 6}, table: []int{0, 1, 2}, expect: []int{4, 5, 6}},
		{msg: "index{4, 5, 6} by{0, 2, 1} -> index{4, 6, 5}",
			index: []int{4, 5, 6}, table: []int{0, 2, 1}, expect: []int{4, 6, 5}},

		{msg: "index{4, 5, 6} by{1, 0, 2} -> index{5, 4, 6}",
			index: []int{4, 5, 6}, table: []int{1, 0, 2}, expect: []int{5, 4, 6}},
		{msg: "index{4, 5, 6} by{1, 2, 0} -> index{6, 4, 5}",
			index: []int{4, 5, 6}, table: []int{1, 2, 0}, expect: []int{6, 4, 5}},

		{msg: "index{4, 5, 6} by{2, 0, 1} -> index{5, 6, 4}",
			index: []int{4, 5, 6}, table: []int{2, 0, 1}, expect: []int{5, 6, 4}},
		{msg: "index{4, 5, 6} by{2, 1, 0} -> index{6, 5, 4}",
			index: []int{4, 5, 6}, table: []int{2, 1, 0}, expect: []int{6, 5, 4}},
	}
	for _, c := range cases {
		actual := indexConvert(c.index, c.table)
		if !reflect.DeepEqual(c.expect, actual) {
			t.Fatalf("(%s) expect %v but actual %v", c.msg, c.expect, actual)
		}
	}
}

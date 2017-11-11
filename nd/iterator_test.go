package nd

import (
	"reflect"
	"testing"
)

func TestArrayIterator(t *testing.T) {
	cases := []struct {
		expect []int
	}{
		{[]int{0, 0, 0}},
		{[]int{0, 0, 1}},
		{[]int{0, 0, 2}},
		{[]int{0, 1, 0}},
		{[]int{0, 1, 1}},
		{[]int{0, 1, 2}},
		{[]int{1, 0, 0}},
		{[]int{1, 0, 1}},
		{[]int{1, 0, 2}},
		{[]int{1, 1, 0}},
		{[]int{1, 1, 1}},
		{[]int{1, 1, 2}},
	}

	itr := NewShape(2, 2, 3).Iterator()
	if !itr.OK() {
		t.Fail()
	}
	for _, c := range cases {
		if !itr.OK() {
			t.Fatalf("not ok at %v", c.expect)
		}
		if !reflect.DeepEqual(itr.Index(), c.expect) {
			t.Fatalf("expect %v got %v", c.expect, itr.Index())
		}
		itr.Next()
	}
	if itr.OK() {
		t.Fail()
	}
}

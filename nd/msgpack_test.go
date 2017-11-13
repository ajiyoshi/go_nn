package nd

import (
	"os"
	"testing"
)

func TestHoge(t *testing.T) {
	cases := []struct {
		msg    string
		path   string
		expect Array
	}{
		{
			msg:  "float64 (3, 2)",
			path: "./t/float_3_2.mp",
			expect: NewArray(NewShape(3, 2), []float64{
				1, 2,
				3, 4,
				5, 6,
			}),
		},
		{
			msg:  "int (2, 3)",
			path: "./t/int_2_3.mp",
			expect: NewArray(NewShape(2, 3), []float64{
				1, 2, 3,
				4, 5, 6,
			}),
		},
	}
	for _, c := range cases {
		f, err := os.Open(c.path)
		if err != nil {
			t.Fatal(err)
		}

		actual, err := NewDecodedArray(f)
		if err != nil {
			t.Fatal(err)
		}
		if !actual.Equals(c.expect) {
			t.Fatalf("(%s) expect %s got %s", c.msg, c.expect, actual)
		}
	}

}

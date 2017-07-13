package gocnn

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestCol2im(t *testing.T) {
	shape := func(n, ch, r, c int) ImageShape {
		return ImageShape{n: n, ch: ch, row: r, col: c}
	}
	cases := []struct {
		title  string
		shape  ImageShape
		image  []float64
		col    mat64.Matrix
		expect []float64
		pad    int
		stride int
		fr     int
		fc     int
	}{
		{
			title: "col2im",
			shape: shape(2, 1, 3, 3),
			image: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				9, 8, 7,
				6, 5, 4,
				3, 2, 1,
			},
			col: mat64.NewDense(2, 9, []float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				9, 8, 7, 6, 5, 4, 3, 2, 1,
			}),
			expect: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				9, 8, 7,
				6, 5, 4,
				3, 2, 1,
			},
			fr: 3, fc: 3,
			stride: 1,
			pad:    0,
		},
		{
			title: "col2im",
			shape: shape(1, 2, 3, 3),
			image: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				10, 20, 30,
				40, 50, 60,
				70, 80, 90,
			},
			col: mat64.NewDense(4, 8, []float64{
				1, 2, 4, 5, 10, 20, 40, 50,
				2, 3, 5, 6, 20, 30, 50, 60,
				4, 5, 7, 8, 40, 50, 70, 80,
				5, 6, 8, 9, 50, 60, 80, 90,
			}),
			expect: []float64{
				1, 4, 3,
				8, 20, 12,
				7, 16, 9,

				10, 40, 30,
				80, 200, 120,
				70, 160, 90,
			},
			fr: 2, fc: 2,
			stride: 1,
			pad:    0,
		},
		{
			title: "im2col",
			shape: shape(1, 2, 2, 2),
			image: []float64{
				1, 2,
				3, 4,

				5, 6,
				7, 8,
			},
			col: mat64.NewDense(9, 8, []float64{
				0, 0, 0, 1, 0, 0, 0, 5,
				0, 0, 1, 2, 0, 0, 5, 6,
				0, 0, 2, 0, 0, 0, 6, 0,
				0, 1, 0, 3, 0, 5, 0, 7,
				1, 2, 3, 4, 5, 6, 7, 8,
				2, 0, 4, 0, 6, 0, 8, 0,
				0, 3, 0, 0, 0, 7, 0, 0,
				3, 4, 0, 0, 7, 8, 0, 0,
				4, 0, 0, 0, 8, 0, 0, 0,
			}),
			expect: []float64{
				4, 8,
				12, 16,

				20, 24,
				28, 32,
			},
			fr: 2, fc: 2,
			stride: 1,
			pad:    1,
		},

		{
			title: "im2col",
			shape: shape(2, 1, 2, 2),
			image: []float64{
				1, 2,
				3, 4,

				5, 6,
				7, 8,
			},
			col: mat64.NewDense(18, 4, []float64{
				0, 0, 0, 1,
				0, 0, 1, 2,
				0, 0, 2, 0,
				0, 1, 0, 3,
				1, 2, 3, 4,
				2, 0, 4, 0,
				0, 3, 0, 0,
				3, 4, 0, 0,
				4, 0, 0, 0,
				0, 0, 0, 5,
				0, 0, 5, 6,
				0, 0, 6, 0,
				0, 5, 0, 7,
				5, 6, 7, 8,
				6, 0, 8, 0,
				0, 7, 0, 0,
				7, 8, 0, 0,
				8, 0, 0, 0,
			}),
			expect: []float64{
				4, 8,
				12, 16,

				20, 24,
				28, 32,
			},
			fr: 2, fc: 2,
			stride: 1,
			pad:    1,
		},
	}

	for _, c := range cases {
		image := NewImages(c.shape, c.image)
		col := Im2col(image, c.fr, c.fc, c.stride, c.pad)
		if !mat64.EqualApprox(col, c.col, 0.01) {
			t.Fatalf("%s expect \n%.2g but got \n%.2g\n",
				c.title, mat64.Formatted(c.col), mat64.Formatted(col))
		}

		actual := Col2im(c.col, &c.shape, c.fr, c.fc, c.stride, c.pad)

		expect := NewImages(c.shape, c.expect)
		if !actual.Equal(expect) {
			t.Fatalf("%s expect \n%v but got \n%v\n",
				c.title, actual, expect)
		}
	}
}

func TestImageMatrix(t *testing.T) {
	shape := func(n, ch, r, c int) ImageShape {
		return ImageShape{n: n, ch: ch, row: r, col: c}
	}
	cases := []struct {
		title  string
		shape  ImageShape
		image  []float64
		expect mat64.Matrix
	}{
		{
			title: "image matrix",
			shape: shape(2, 1, 3, 3),
			image: []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,

				9, 8, 7,
				6, 5, 4,
				3, 2, 1,
			},
			expect: mat64.NewDense(2, 9, []float64{
				1, 2, 3, 4, 5, 6, 7, 8, 9,
				9, 8, 7, 6, 5, 4, 3, 2, 1,
			}),
		},
		{
			title: "image matrix",
			shape: shape(1, 2, 2, 2),
			image: []float64{
				1, 2,
				3, 4,

				5, 6,
				7, 8,
			},
			expect: mat64.NewDense(1, 8, []float64{
				1, 2, 3, 4, 5, 6, 7, 8,
			}),
		},
	}

	for _, c := range cases {
		image := NewImages(c.shape, c.image)
		actual := image.Matrix()
		if !mat64.EqualApprox(actual, c.expect, 0.01) {
			t.Fatalf("%s expect \n%.2g but got \n%.2g\n",
				c.title, mat64.Formatted(c.expect), mat64.Formatted(actual))
		}

	}
}

func TestImageString(t *testing.T) {
	shape := func(n, ch, r, c int) ImageShape {
		return ImageShape{n: n, ch: ch, row: r, col: c}
	}
	cases := []struct {
		title  string
		input  *SimpleStrage
		expect string
	}{
		{
			title: "image matrix string",
			input: NewImages(shape(1, 2, 2, 3), []float64{
				1, 2, 3,
				4, 5, 6,

				7, 8, 9,
				8, 7, 6,
			}),
			expect: `{
⎡1  2  3⎤
⎣4  5  6⎦
⎡7  8  9⎤
⎣8  7  6⎦
},
`,
		},
	}

	for _, c := range cases {
		actual := c.input.String()
		if actual != c.expect {
			t.Fatalf("%s expect \n%s but got \n%s\n",
				c.title, c.expect, actual)
		}
	}
}

func TestReshape(t *testing.T) {
	shape := func(n, ch, r, c int) ImageShape {
		return ImageShape{n: n, ch: ch, row: r, col: c}
	}
	cases := []struct {
		title  string
		shape  ImageShape
		image  mat64.Matrix
		expect string
	}{
		{
			title: "image matrix",
			image: mat64.NewDense(4, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				8, 7, 6,
			}),
			shape: shape(1, 2, 2, 3),
			expect: `{
⎡1  2  3⎤
⎣4  5  6⎦
⎡7  8  9⎤
⎣8  7  6⎦
},
`,
		},
		{
			title: "image matrix",
			image: mat64.NewDense(4, 3, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				8, 7, 6,
			}),
			shape: shape(2, 1, 2, 3),
			expect: `{
⎡1  2  3⎤
⎣4  5  6⎦
},
{
⎡7  8  9⎤
⎣8  7  6⎦
},
`,
		},

		{
			title: "image matrix",
			image: mat64.NewDense(9, 2, []float64{
				0.1, 0.2,
				0.3, 0.4,
				0.5, 0.6,
				0.7, 0.8,
				0.9, 1.0,
				1.1, 1.2,
				1.3, 1.4,
				1.5, 1.6,
				1.6, 1.7,
			}),
			shape: shape(1, 3, 3, 2),
			expect: `{
⎡0.1  0.2⎤
⎢0.3  0.4⎥
⎣0.5  0.6⎦
⎡0.7  0.8⎤
⎢0.9    1⎥
⎣1.1  1.2⎦
⎡1.3  1.4⎤
⎢1.5  1.6⎥
⎣1.6  1.7⎦
},
`,
		},
		{
			title: "image matrix",
			image: mat64.NewDense(9, 2, []float64{
				0.1, 0.2,
				0.3, 0.4,
				0.5, 0.6,
				0.7, 0.8,
				0.9, 1.0,
				1.1, 1.2,
				1.3, 1.4,
				1.5, 1.6,
				1.6, 1.7,
			}),
			shape: shape(3, 3, 1, 2),
			expect: `{
[0.1  0.2]
[0.3  0.4]
[0.5  0.6]
},
{
[0.7  0.8]
[0.9    1]
[1.1  1.2]
},
{
[1.3  1.4]
[1.5  1.6]
[1.6  1.7]
},
`,
		},
	}

	for _, c := range cases {
		actual := NewReshaped(c.shape, c.image).String()
		if actual != c.expect {
			t.Fatalf("%s expect \n%s but got \n%s\n",
				c.title, c.expect, actual)
		}
	}
}

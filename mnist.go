package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"golang.org/x/exp/mmap"
	"image"
	"image/color"
	"image/png"
	"io"
)

type Mnist struct {
	Images *MnistImage
	Labels *MnistLabel
	index  int
}

type MnistImage struct {
	m           *mmap.ReaderAt
	buf         []byte
	MagicNumber int
	Num         int
	Rows        int
	Cols        int
}

type MnistLabel struct {
	m           *mmap.ReaderAt
	MagicNumber int
	Num         int
}

func NewMnist(image, label string) (*Mnist, error) {
	mi, err := NewMnistImage(image)
	if err != nil {
		return nil, err
	}
	ml, err := NewMnistLabel(label)
	if err != nil {
		mi.Close()
		return nil, err
	}
	return &Mnist{
		Images: mi,
		Labels: ml,
	}, nil
}

func (m *Mnist) Close() {
	m.Images.Close()
	m.Labels.Close()
}

func (m *Mnist) Jump(i int) error {
	err := m.Images.Jump(i)
	if err != nil {
		return err
	}
	m.index = i
	return nil
}
func (m *Mnist) Label() byte {
	return m.Labels.At(m.index)
}
func (m *Mnist) Image() []byte {
	return m.Images.Buf()
}
func (m *Mnist) At(i int) ([]byte, byte) {
	m.Jump(i)
	return m.Image(), m.Label()
}

func (m *Mnist) DumpPng(w io.Writer) error {
	return m.Images.DumpPng(w)
}

func NewMnistImage(path string) (*MnistImage, error) {
	m, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}

	ret, err := initMnistImage(m)
	if err != nil {
		m.Close()
		return nil, err
	}

	return ret, nil
}

func initMnistImage(m *mmap.ReaderAt) (*MnistImage, error) {
	magic, err := Int32At(m, 0)
	if err != nil {
		return nil, err
	}
	num, err := Int32At(m, 4)
	if err != nil {
		return nil, err
	}
	rows, err := Int32At(m, 8)
	if err != nil {
		return nil, err
	}
	cols, err := Int32At(m, 12)
	if err != nil {
		return nil, err
	}

	return &MnistImage{
		m:           m,
		buf:         make([]byte, rows*cols),
		MagicNumber: int(magic),
		Num:         int(num),
		Rows:        int(rows),
		Cols:        int(cols),
	}, nil
}

func (m *MnistImage) Close() error {
	return m.m.Close()
}

func (m *MnistImage) Jump(i int) error {
	offset := 16 + int64(i)*int64(m.Rows)*int64(m.Cols)
	err := MustRead(m.m, offset, m.buf)
	if err != nil {
		return err
	}
	return nil
}
func (m *MnistImage) Buf() []byte {
	return m.buf
}

func (m *MnistImage) DumpPng(w io.Writer) error {
	img := image.NewGray(image.Rect(0, 0, m.Rows, m.Cols))
	for x := 0; x < m.Rows; x++ {
		for y := 0; y < m.Cols; y++ {
			i := x + y*m.Rows
			img.Set(x, y, color.Gray{m.buf[i]})
		}
	}
	return png.Encode(w, img)
}

func NewMnistLabel(path string) (*MnistLabel, error) {
	m, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}

	magic, err := Int32At(m, 0)
	if err != nil {
		return nil, err
	}
	num, err := Int32At(m, 4)
	if err != nil {
		return nil, err
	}

	return &MnistLabel{
		m:           m,
		MagicNumber: int(magic),
		Num:         int(num),
	}, nil
}

func (m *MnistLabel) Close() error {
	return m.m.Close()
}

func (m *MnistLabel) At(i int) byte {
	offset := 8 + i
	return m.m.At(offset)
}

func Int32At(m *mmap.ReaderAt, offset int64) (int32, error) {
	buf := make([]byte, 4)
	err := MustRead(m, offset, buf)
	if err != nil {
		return 0, err
	}
	r := bytes.NewReader(buf)
	var ret int32
	err = binary.Read(r, binary.BigEndian, &ret)
	if err != nil {
		return 0, err
	}
	return ret, nil
}

func MustRead(m *mmap.ReaderAt, offset int64, buf []byte) error {
	n, err := m.ReadAt(buf, offset)
	if err != nil {
		return err
	}
	if n != len(buf) {
		panic(fmt.Sprintf("bad file(try to read %d bytes but got %d bytes", len(buf), n))
	}
	return nil
}

package mnist

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

/*
Mnist ラベルとイメージをセットにした構造体
*/
type Mnist struct {
	Images *Image
	Labels *Label
	index  int
}

/*
Image イメージ構造体
*/
type Image struct {
	m           *mmap.ReaderAt
	buf         []byte
	MagicNumber int
	Num         int
	Rows        int
	Cols        int
}

/*
Label ラベル構造体
*/
type Label struct {
	m           *mmap.ReaderAt
	MagicNumber int
	Num         int
}

/*
NewMnist イメージとラベルのインデクスファイルを開く
*/
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

/*
Close mmap を閉じる
*/
func (m *Mnist) Close() {
	m.Images.Close()
	m.Labels.Close()
}

/*
Jump i番目のイメージをロードする。 Mnist.Image() Mnist.Label() で対応するイメージやラベルを取得できるようになる
*/
func (m *Mnist) Jump(i int) error {
	err := m.Images.Jump(i)
	if err != nil {
		return err
	}
	m.index = i
	return nil
}

/*
Label Mnist.Jump() でロードしたラベル
*/
func (m *Mnist) Label() byte {
	return m.Labels.At(m.index)
}

/*
Image Mnist.Jump() でロードしたイメージ
*/
func (m *Mnist) Image() []byte {
	return m.Images.Buf()
}

/*
At i番目のイメージとラベルを返す
*/
func (m *Mnist) At(i int) ([]byte, byte) {
	m.Jump(i)
	return m.Image(), m.Label()
}

/*
DumpPng PNG形式でイメージを書き出す
*/
func (m *Mnist) DumpPng(w io.Writer) error {
	return m.Images.DumpPng(w)
}

/*
NewMnistImage Image オブジェクトをロードする。使い終わったら Close() すること
*/
func NewMnistImage(path string) (*Image, error) {
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

/*
initMnistImage マジックナンバー、レコード数、イメージの縦横サイズをロード
*/
func initMnistImage(m *mmap.ReaderAt) (*Image, error) {
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

	return &Image{
		m:           m,
		buf:         make([]byte, rows*cols),
		MagicNumber: int(magic),
		Num:         int(num),
		Rows:        int(rows),
		Cols:        int(cols),
	}, nil
}

/*
Close mmapを閉じる
*/
func (m *Image) Close() error {
	return m.m.Close()
}

/*
Jump i番目のイメージをロード
*/
func (m *Image) Jump(i int) error {
	offset := 16 + int64(i)*int64(m.Rows)*int64(m.Cols)
	err := MustRead(m.m, offset, m.buf)
	if err != nil {
		return err
	}
	return nil
}

/*
Buf Image.Jump でロードした場所のイメージを返す
*/
func (m *Image) Buf() []byte {
	return m.buf
}

/*
DumpPng Image.Jump でロードした場所のイメージをPNG形式で書き出す
*/
func (m *Image) DumpPng(w io.Writer) error {
	img := image.NewGray(image.Rect(0, 0, m.Rows, m.Cols))
	for x := 0; x < m.Rows; x++ {
		for y := 0; y < m.Cols; y++ {
			i := x + y*m.Rows
			img.Set(x, y, color.Gray{m.buf[i]})
		}
	}
	return png.Encode(w, img)
}

/*
NewMnistLabel ラベルファイルを開く
*/
func NewMnistLabel(path string) (*Label, error) {
	m, err := mmap.Open(path)
	if err != nil {
		return nil, err
	}

	ret, err := initMnistLabel(m)
	if err != nil {
		m.Close()
		return nil, err
	}

	return ret, nil
}

func initMnistLabel(m *mmap.ReaderAt) (*Label, error) {
	magic, err := Int32At(m, 0)
	if err != nil {
		return nil, err
	}
	num, err := Int32At(m, 4)
	if err != nil {
		return nil, err
	}

	return &Label{
		m:           m,
		MagicNumber: int(magic),
		Num:         int(num),
	}, nil
}

/*
Close ラベルファイルを閉じる
*/
func (m *Label) Close() error {
	return m.m.Close()
}

/*
At i番目のラベルを取得
*/
func (m *Label) At(i int) byte {
	offset := 8 + i
	return m.m.At(offset)
}

/*
Int32At offsetバイト目から BigEndian で int32 をロードする
*/
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

/*
MustRead mmap.ReaderAt の offset バイト目から len(buf) 分をロード。len(buf) 分読めなければエラー
*/
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

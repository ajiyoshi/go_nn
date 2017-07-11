package mnist

import (
	"bytes"
	"fmt"
	"github.com/gonum/matrix/mat64"
	"io"
	"math/rand"
)

/*
TrainBuffer 学習用に読み込むバッファ
*/
type TrainBuffer struct {
	rows int
	xCol int
	tCol int
	x    []float64
	t    []float64
}

/*
NewTrainBuffer 学習用バッファを初期化。高速化のために連続領域を確保する。
rows バッチ学習のために読み込む行数
xCol 入力1つあたりの次元(mnistなら728)
tCol ラベル1つあたりの次元(mnistなら10)
*/
func NewTrainBuffer(rows, xCol, tCol int) *TrainBuffer {
	return &TrainBuffer{
		rows: rows,
		xCol: xCol,
		tCol: tCol,
		x:    make([]float64, rows*xCol),
		t:    make([]float64, rows*tCol),
	}
}

/*
LoadX 入力データをバッファにコピー
*/
func (buf *TrainBuffer) LoadX(i int, x []byte) error {
	if len(x) != buf.xCol {
		return fmt.Errorf("bad size of data expect:%d but got %d", buf.xCol, len(x))
	}
	offset := i * buf.xCol
	LoadVec(x, buf.x[offset:])
	return nil
}

/*
LoadT ラベルをバッファにコピー
*/
func (buf *TrainBuffer) LoadT(i int, t byte) {
	offset := i * buf.tCol
	copy(buf.t[offset:], labels[t])
}

/*
Load Mnist からイメージとラベルを読み取ってバッファにコピー
*/
func (buf *TrainBuffer) Load(m *Mnist, at []int) {
	rows := len(at)
	for i := 0; i < rows; i++ {
		n := at[i]
		x, t := m.At(n)
		buf.LoadX(i, x)
		buf.LoadT(i, t)
	}
}

/*
Dump デバッグ用に、ロードしているMNISTイメージをアスキーアートにして表示
*/
func (buf *TrainBuffer) Dump(w io.Writer) {
	for i := 0; i < buf.rows; i++ {
		offset := buf.xCol * i
		fmt.Fprintf(w, "%s\n", XToString(buf.x[offset:]))
	}
}

/*
Bake ロードしているイメージとラベルを行列に変換
*/
func (buf *TrainBuffer) Bake() (x, t mat64.Matrix) {
	x = mat64.NewDense(buf.rows, buf.xCol, buf.x)
	t = mat64.NewDense(buf.rows, buf.tCol, buf.t)
	return x, t
}

/*
DumpX MNISTイメージとラベルをセットで書き出す
*/
func DumpX(w io.Writer, x, t mat64.Matrix, i int) {
	xbuf := mat64.Row(nil, i, x)
	tbuf := mat64.Row(nil, i, t)
	fmt.Fprintf(w, "%d:\n%s\n", LabelAsNum(tbuf), XToString(xbuf))
}

/*
LabelAsNum ラベルを整数として返す
*/
func LabelAsNum(v []float64) int {
	max := v[0]
	ret := 0

	for i := 1; i < len(v); i++ {
		if v[i] > max {
			max = v[i]
			ret = i
		}
	}
	return ret
}

func Seq(x, n int) []int {
	ret := make([]int, n)
	for i := 0; i < n; i++ {
		ret[i] = x + i
	}
	return ret
}

func RandamSeq(n, max int) []int {
	ret := make([]int, n)
	for i := 0; i < n; i++ {
		ret[i] = rand.Intn(max)
	}
	return ret
}

func GetX(data []float64, x, y int) float64 {
	return data[x+y*28]
}
func AddHoge(data []float64, v float64, x, y int) {
	data[(x/4)+(y/4)*7] += v
}
func GetHoge(data []float64, x, y int) float64 {
	return data[x+7*y]
}

func XToString(data []float64) string {
	w := bytes.NewBuffer(make([]byte, 0, 50))
	buf := make([]float64, 49)

	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			s := GetX(data, x, y)
			AddHoge(buf, s, x, y)
		}
	}
	for y := 0; y < 7; y++ {
		for x := 0; x < 7; x++ {
			if GetHoge(buf, x, y) > 5 {
				w.WriteString("x")
			} else {
				w.WriteString(".")
			}
		}
		w.WriteString("\n")
	}
	return w.String()
}

func LoadVec(raw []byte, buf []float64) {
	for i, v := range raw {
		buf[i] = float64(v) / 255.0
	}
}

var labels [][]float64 = [][]float64{
	[]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
	[]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
}

func LoadLabel(label byte) *mat64.Vector {
	return mat64.NewVector(10, labels[label])
}

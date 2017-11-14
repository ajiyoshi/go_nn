package nd

import (
	"bytes"
	"encoding/binary"
	"github.com/vmihailenco/msgpack"
	"io"
	"os"
	"strings"
)

type numpyMsgpack struct {
	Type  string `msgpack:"type"`
	Data  []byte `msgpack:"data"`
	Shape []int  `msgpack:"shape"`
}

func Must(a Array, err error) Array {
	if err != nil {
		panic(err)
	}
	return a
}

func Load(path string) (Array, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return NewDecodedArray(f)
}

func NewDecodedArray(r io.Reader) (Array, error) {
	var buf numpyMsgpack
	if err := buf.Decode(r); err != nil {
		return nil, err
	}
	return buf.Extract()
}

func (x *numpyMsgpack) Decode(i io.Reader) error {
	d := msgpack.NewDecoder(i)
	if err := d.Decode(x); err != nil {
		return err
	}
	return nil
}

func (x *numpyMsgpack) Extract() (Array, error) {
	s := NewShape(x.Shape...)
	ret := make([]float64, s.Size())
	err := extract(ret, bytes.NewReader(x.Data), x.Decoder())
	if err != nil {
		return nil, err
	}
	return NewArray(s, ret), nil
}

func (x *numpyMsgpack) Decoder() Decoder {
	if strings.HasPrefix(x.Type, "<f4") {
		return f32Decoder
	} else if strings.HasPrefix(x.Type, "<f") {
		return floatDecoder
	} else if strings.HasPrefix(x.Type, "<i") {
		return intDecoder
	}
	panic("unknown dtype")
}

type Decoder func(io.Reader, *float64) error

func floatDecoder(r io.Reader, ret *float64) error {
	return binary.Read(r, binary.LittleEndian, ret)
}
func intDecoder(r io.Reader, ret *float64) error {
	var i int64
	err := binary.Read(r, binary.LittleEndian, &i)
	if err == nil {
		*ret = float64(i)
	}
	return err
}
func f32Decoder(r io.Reader, ret *float64) error {
	var f float32
	err := binary.Read(r, binary.LittleEndian, &f)
	if err == nil {
		*ret = float64(f)
	}
	return err
}

func extract(buf []float64, r io.Reader, decode Decoder) error {
	for i := 0; i < len(buf); i++ {
		err := decode(r, &buf[i])
		if err != nil {
			return err
		}
	}
	return nil
}

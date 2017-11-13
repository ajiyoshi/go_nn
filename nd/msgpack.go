package nd

import (
	"bytes"
	"encoding/binary"
	"github.com/vmihailenco/msgpack"
	"io"
	"strings"
)

type numpyMsgpack struct {
	Type  string `msgpack:"type"`
	Data  []byte `msgpack:"data"`
	Shape []int  `msgpack:"shape"`
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
	if strings.HasPrefix(x.Type, "<f") {
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

func extract(buf []float64, r io.Reader, decode Decoder) error {
	for i := 0; i < len(buf); i++ {
		err := decode(r, &buf[i])
		if err != nil {
			return err
		}
	}
	return nil
}

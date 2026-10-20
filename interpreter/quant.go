package interpreter

import "math"

// QuantMultiplier holds a decomposed scale factor as (multiplier, shift)
// for efficient fixed-point arithmetic.
type QuantMultiplier struct {
	Multiplier int32
	Shift      int
}

// QuantizeMultiplier decomposes a float64 scale into a fixed-point
// multiplier (Q0.31 format) and an exponent shift. Like TFLite, it flushes
// scales below 2^-32 to zero and saturates scales of 2^31 and above.
func QuantizeMultiplier(realMultiplier float64) QuantMultiplier {
	if realMultiplier == 0 {
		return QuantMultiplier{0, 0}
	}
	sign := 1
	if realMultiplier < 0 {
		sign = -1
		realMultiplier = -realMultiplier
	}
	shift := 0
	for realMultiplier < 0.5 {
		realMultiplier *= 2.0
		shift--
	}
	for realMultiplier >= 1.0 {
		realMultiplier /= 2.0
		shift++
	}
	qMul := int64(math.Round(realMultiplier * float64(int64(1)<<31)))
	if qMul == int64(1)<<31 {
		qMul /= 2
		shift++
	}
	if shift < -31 {
		return QuantMultiplier{0, 0}
	}
	if shift > 30 {
		return QuantMultiplier{math.MaxInt32 * int32(sign), 30}
	}
	return QuantMultiplier{int32(qMul) * int32(sign), shift}
}

// MultiplyByQuantizedMultiplier computes x * multiplier * 2^shift with
// TFLite's reference rounding: a rounding doubling high multiply followed by
// a rounding right shift. Matching that double rounding bit for bit keeps
// streaming models from drifting away from TFLite over many invocations.
func MultiplyByQuantizedMultiplier(x int32, qm QuantMultiplier) int32 {
	leftShift, rightShift := 0, 0
	if qm.Shift > 0 {
		leftShift = qm.Shift
	} else {
		rightShift = -qm.Shift
	}
	x = saturateInt32(int64(x) << leftShift)
	return roundingDivideByPOT(saturatingRoundingDoublingHighMul(x, qm.Multiplier), rightShift)
}

// saturatingRoundingDoublingHighMul returns round(a*b / 2^31), saturating
// the single overflowing case.
func saturatingRoundingDoublingHighMul(a, b int32) int32 {
	if a == math.MinInt32 && b == math.MinInt32 {
		return math.MaxInt32
	}
	ab := int64(a) * int64(b)
	nudge := int64(1 << 30)
	if ab < 0 {
		nudge = 1 - (1 << 30)
	}
	return int32((ab + nudge) / (1 << 31))
}

// roundingDivideByPOT returns x / 2^exp rounded half away from zero.
func roundingDivideByPOT(x int32, exp int) int32 {
	mask := int32(1)<<exp - 1
	rem := x & mask
	th := mask >> 1
	if x < 0 {
		th++
	}
	q := x >> exp
	if rem > th {
		q++
	}
	return q
}

func saturateInt32(v int64) int32 {
	if v > math.MaxInt32 {
		return math.MaxInt32
	}
	if v < math.MinInt32 {
		return math.MinInt32
	}
	return int32(v)
}

// ClampInt8 clamps a value to int8 range.
func ClampInt8(v int32) int8 {
	if v < -128 {
		return -128
	}
	if v > 127 {
		return 127
	}
	return int8(v)
}

// ClampUint8 clamps a value to uint8 range.
func ClampUint8(v int32) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// Logistic computes 1/(1+exp(-x)).
func Logistic(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

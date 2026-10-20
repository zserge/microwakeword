package audiofrontend

import "math"

// log1pf is glibc's (fdlibm's) single-precision log1pf, sysdeps/ieee754/
// flt-32/s_log1pf.c, evaluated in float32 step by step. The filterbank
// weights are float32 functions of log1pf, and glibc's result is not always
// correctly rounded: float32(math.Log1p(x)) differs from it by one ulp on
// about 10% of the filterbank's bins, which moves two of the Q12 weights by
// one and the features with them. The reference (pymicro-features, which
// generated microWakeWord's training features) links glibc's libm, so this
// is the function the reference weights come from.
//
// Every product is wrapped in float32() so the compiler cannot fuse it into
// an FMA on any architecture.
func log1pf(x float32) float32 {
	const (
		ln2Hi = float32(6.9313812256e-01) // 0x3f317180
		ln2Lo = float32(9.0580006145e-06) // 0x3717f7d1
		lp1   = float32(6.6666668653e-01) // 0x3F2AAAAB
		lp2   = float32(4.0000000596e-01) // 0x3ECCCCCD
		lp3   = float32(2.8571429849e-01) // 0x3E924925
		lp4   = float32(2.2222198546e-01) // 0x3E638E29
		lp5   = float32(1.8183572590e-01) // 0x3E3A3325
		lp6   = float32(1.5313838422e-01) // 0x3E1CD04F
		lp7   = float32(1.4798198640e-01) // 0x3E178897
	)
	hx := int32(math.Float32bits(x))
	ax := hx & 0x7fffffff

	var f, c float32
	var hu int32
	k := int32(1)
	if hx < 0x3ed413d7 { // x < 0.41422
		if ax >= 0x3f800000 { // x <= -1.0
			if x == -1 {
				return float32(math.Inf(-1))
			}
			return float32(math.NaN())
		}
		if ax < 0x31000000 { // |x| < 2**-29
			if ax < 0x24800000 { // |x| < 2**-54
				return x
			}
			return x - float32(float32(x*x)*0.5)
		}
		if hx > 0 || hx <= int32(-0x416a09e1) { // -0.2929 < x < 0.41422 (0xbe95f61f)
			k, f, hu = 0, x, 1
		}
	}
	if hx >= 0x7f800000 {
		return x + x
	}
	if k != 0 {
		var u float32
		if hx < 0x5a000000 {
			u = 1 + x
			hu = int32(math.Float32bits(u))
			k = (hu >> 23) - 127
			// correction term
			if k > 0 {
				c = 1 - (u - x)
			} else {
				c = x - (u - 1)
			}
			c /= u
		} else {
			u = x
			hu = int32(math.Float32bits(u))
			k = (hu >> 23) - 127
			c = 0
		}
		hu &= 0x007fffff
		if hu < 0x3504f7 {
			u = math.Float32frombits(uint32(hu | 0x3f800000)) // normalize u
		} else {
			k++
			u = math.Float32frombits(uint32(hu | 0x3f000000)) // normalize u/2
			hu = (0x00800000 - hu) >> 2
		}
		f = u - 1
	}
	kf := float32(k)
	hfsq := float32(float32(0.5*f) * f)
	if hu == 0 { // |f| < 2**-20
		if f == 0 {
			if k == 0 {
				return 0
			}
			c += float32(kf * ln2Lo)
			return float32(kf*ln2Hi) + c
		}
		r := float32(hfsq * (1.0 - float32(float32(0.66666666666666666)*f)))
		if k == 0 {
			return f - r
		}
		return float32(kf*ln2Hi) - ((r - (float32(kf*ln2Lo) + c)) - f)
	}
	s := f / (2 + f)
	z := float32(s * s)
	r := float32(z * lp7)
	r = float32(z * (lp6 + r))
	r = float32(z * (lp5 + r))
	r = float32(z * (lp4 + r))
	r = float32(z * (lp3 + r))
	r = float32(z * (lp2 + r))
	r = float32(z * (lp1 + r))
	if k == 0 {
		return f - (hfsq - float32(s*(hfsq+r)))
	}
	return float32(kf*ln2Hi) - ((hfsq - (float32(s*(hfsq+r)) + (float32(kf*ln2Lo) + c))) - f)
}

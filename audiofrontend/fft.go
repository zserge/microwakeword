package audiofrontend

import "math"

// This file is a port of KISS FFT (kiss_fft.c and tools/kiss_fftr.c) built
// with FIXED_POINT=16, as vendored by the TFLite Micro microfrontend. Every
// rounding and wrap-around of the 16-bit arithmetic is kept as in the C code:
// sums are stored into int16 (wrapping), products are int32 and rounded by
// sround, and HALF_OF in the real-FFT post-processing shifts the int-promoted
// sum before truncating.

type cpx struct{ r, i int16 }

const sampMax = 32767

// sround is ((x + (1<<(FRACBITS-1))) >> FRACBITS) cast to kiss_fft_scalar.
func sround(x int32) int16 { return int16((x + (1 << 14)) >> 15) }

func smul(a, b int16) int32 { return int32(a) * int32(b) }

func cmul(a, b cpx) cpx {
	return cpx{
		r: sround(smul(a.r, b.r) - smul(a.i, b.i)),
		i: sround(smul(a.r, b.i) + smul(a.i, b.r)),
	}
}

// fixdiv is C_FIXDIV: multiply both parts by SAMP_MAX/div with rounding.
func fixdiv(c *cpx, div int16) {
	s := int16(sampMax / int32(div))
	c.r = sround(smul(c.r, s))
	c.i = sround(smul(c.i, s))
}

func cexp(phase float64) cpx {
	return cpx{
		// The explicit float64 rounds the product, so arm64 cannot fuse it with the add.
		r: int16(math.Floor(.5 + float64(sampMax*math.Cos(phase)))),
		i: int16(math.Floor(.5 + float64(sampMax*math.Sin(phase)))),
	}
}

// kissFFT is a forward complex FFT state (kiss_fft_state, inverse = 0).
type kissFFT struct {
	nfft     int
	factors  []int
	twiddles []cpx
}

func newKissFFT(nfft int) *kissFFT {
	st := &kissFFT{nfft: nfft, twiddles: make([]cpx, nfft)}
	const pi = 3.141592653589793238462643383279502884197169399375105820974944
	for i := range nfft {
		phase := -2 * pi * float64(i) / float64(nfft)
		st.twiddles[i] = cexp(phase)
	}
	st.factors = kfFactor(nfft)
	return st
}

// kfFactor factors out powers of 4, then 2, then the remaining primes,
// returning p1, m1, p2, m2, ...
func kfFactor(n int) []int {
	var fac []int
	p := 4
	floorSqrt := math.Floor(math.Sqrt(float64(n)))
	for {
		for n%p != 0 {
			switch p {
			case 4:
				p = 2
			case 2:
				p = 3
			default:
				p += 2
			}
			if float64(p) > floorSqrt {
				p = n
			}
		}
		n /= p
		fac = append(fac, p, n)
		if n <= 1 {
			return fac
		}
	}
}

// transform computes out = FFT(in); in and out must not overlap.
func (st *kissFFT) transform(in, out []cpx) {
	st.work(out, in, 0, 1, st.factors)
}

func (st *kissFFT) work(out, in []cpx, inOff, fstride int, factors []int) {
	p, m := factors[0], factors[1]
	if m == 1 {
		for j := range p {
			out[j] = in[inOff+j*fstride]
		}
	} else {
		for j := range p {
			st.work(out[j*m:], in, inOff+j*fstride, fstride*p, factors[2:])
		}
	}
	switch p {
	case 2:
		st.bfly2(out, fstride, m)
	case 3:
		st.bfly3(out, fstride, m)
	case 4:
		st.bfly4(out, fstride, m)
	case 5:
		st.bfly5(out, fstride, m)
	default:
		st.bflyGeneric(out, fstride, m, p)
	}
}

func (st *kissFFT) bfly2(f []cpx, fstride, m int) {
	tw := st.twiddles
	for k := range m {
		a, b := &f[k], &f[k+m]
		fixdiv(a, 2)
		fixdiv(b, 2)
		t := cmul(*b, tw[k*fstride])
		b.r, b.i = a.r-t.r, a.i-t.i
		a.r += t.r
		a.i += t.i
	}
}

func (st *kissFFT) bfly4(f []cpx, fstride, m int) {
	tw := st.twiddles
	m2, m3 := 2*m, 3*m
	for k := range m {
		f0, f1, f2, f3 := &f[k], &f[k+m], &f[k+m2], &f[k+m3]
		fixdiv(f0, 4)
		fixdiv(f1, 4)
		fixdiv(f2, 4)
		fixdiv(f3, 4)
		s0 := cmul(*f1, tw[k*fstride])
		s1 := cmul(*f2, tw[2*k*fstride])
		s2 := cmul(*f3, tw[3*k*fstride])
		s5 := cpx{f0.r - s1.r, f0.i - s1.i}
		f0.r += s1.r
		f0.i += s1.i
		s3 := cpx{s0.r + s2.r, s0.i + s2.i}
		s4 := cpx{s0.r - s2.r, s0.i - s2.i}
		f2.r, f2.i = f0.r-s3.r, f0.i-s3.i
		f0.r += s3.r
		f0.i += s3.i
		f1.r, f1.i = s5.r+s4.i, s5.i-s4.r
		f3.r, f3.i = s5.r-s4.i, s5.i+s4.r
	}
}

func (st *kissFFT) bfly3(f []cpx, fstride, m int) {
	tw := st.twiddles
	m2 := 2 * m
	epi3 := tw[fstride*m]
	for k := range m {
		f0, f1, f2 := &f[k], &f[k+m], &f[k+m2]
		fixdiv(f0, 3)
		fixdiv(f1, 3)
		fixdiv(f2, 3)
		s1 := cmul(*f1, tw[k*fstride])
		s2 := cmul(*f2, tw[2*k*fstride])
		s3 := cpx{s1.r + s2.r, s1.i + s2.i}
		s0 := cpx{s1.r - s2.r, s1.i - s2.i}
		f1.r = f0.r - s3.r>>1
		f1.i = f0.i - s3.i>>1
		s0.r = sround(smul(s0.r, epi3.i))
		s0.i = sround(smul(s0.i, epi3.i))
		f0.r += s3.r
		f0.i += s3.i
		f2.r = f1.r + s0.i
		f2.i = f1.i - s0.r
		f1.r -= s0.i
		f1.i += s0.r
	}
}

func (st *kissFFT) bfly5(f []cpx, fstride, m int) {
	tw := st.twiddles
	ya := tw[fstride*m]
	yb := tw[fstride*2*m]
	smulr := func(a, b int16) int16 { return sround(smul(a, b)) }
	for u := range m {
		f0, f1, f2, f3, f4 := &f[u], &f[u+m], &f[u+2*m], &f[u+3*m], &f[u+4*m]
		fixdiv(f0, 5)
		fixdiv(f1, 5)
		fixdiv(f2, 5)
		fixdiv(f3, 5)
		fixdiv(f4, 5)
		s0 := *f0
		s1 := cmul(*f1, tw[u*fstride])
		s2 := cmul(*f2, tw[2*u*fstride])
		s3 := cmul(*f3, tw[3*u*fstride])
		s4 := cmul(*f4, tw[4*u*fstride])
		s7 := cpx{s1.r + s4.r, s1.i + s4.i}
		s10 := cpx{s1.r - s4.r, s1.i - s4.i}
		s8 := cpx{s2.r + s3.r, s2.i + s3.i}
		s9 := cpx{s2.r - s3.r, s2.i - s3.i}
		f0.r += s7.r + s8.r
		f0.i += s7.i + s8.i
		s5 := cpx{
			s0.r + smulr(s7.r, ya.r) + smulr(s8.r, yb.r),
			s0.i + smulr(s7.i, ya.r) + smulr(s8.i, yb.r),
		}
		s6 := cpx{
			smulr(s10.i, ya.i) + smulr(s9.i, yb.i),
			-smulr(s10.r, ya.i) - smulr(s9.r, yb.i),
		}
		f1.r, f1.i = s5.r-s6.r, s5.i-s6.i
		f4.r, f4.i = s5.r+s6.r, s5.i+s6.i
		s11 := cpx{
			s0.r + smulr(s7.r, yb.r) + smulr(s8.r, ya.r),
			s0.i + smulr(s7.i, yb.r) + smulr(s8.i, ya.r),
		}
		s12 := cpx{
			-smulr(s10.i, yb.i) + smulr(s9.i, ya.i),
			smulr(s10.r, yb.i) - smulr(s9.r, ya.i),
		}
		f2.r, f2.i = s11.r+s12.r, s11.i+s12.i
		f3.r, f3.i = s11.r-s12.r, s11.i-s12.i
	}
}

func (st *kissFFT) bflyGeneric(f []cpx, fstride, m, p int) {
	tw := st.twiddles
	scratch := make([]cpx, p)
	for u := range m {
		k := u
		for q1 := range p {
			scratch[q1] = f[k]
			fixdiv(&scratch[q1], int16(p))
			k += m
		}
		k = u
		for range p {
			twidx := 0
			f[k] = scratch[0]
			for q := 1; q < p; q++ {
				twidx += fstride * k
				if twidx >= st.nfft {
					twidx -= st.nfft
				}
				t := cmul(scratch[q], tw[twidx])
				f[k].r += t.r
				f[k].i += t.i
			}
			k += m
		}
	}
}

// kissFFTR is a forward real FFT of nfft (even) points.
type kissFFTR struct {
	sub           *kissFFT
	in, tmp       []cpx
	superTwiddles []cpx
}

func newKissFFTR(nfft int) *kissFFTR {
	if nfft&1 != 0 {
		panic("audiofrontend: real FFT size must be even")
	}
	ncfft := nfft >> 1
	st := &kissFFTR{
		sub:           newKissFFT(ncfft),
		in:            make([]cpx, ncfft),
		tmp:           make([]cpx, ncfft),
		superTwiddles: make([]cpx, ncfft/2),
	}
	for i := range ncfft / 2 {
		phase := -3.14159265358979323846264338327 * (float64(i+1)/float64(ncfft) + .5)
		st.superTwiddles[i] = cexp(phase)
	}
	return st
}

// transform computes the ncfft+1 bins of the FFT of the nfft real samples in
// timedata into freq.
func (st *kissFFTR) transform(timedata []int16, freq []cpx) {
	ncfft := st.sub.nfft
	for i := range st.in {
		st.in[i] = cpx{timedata[2*i], timedata[2*i+1]}
	}
	st.sub.transform(st.in, st.tmp)
	tmp := st.tmp

	tdc := tmp[0]
	fixdiv(&tdc, 2)
	freq[0] = cpx{tdc.r + tdc.i, 0}
	freq[ncfft] = cpx{tdc.r - tdc.i, 0}

	for k := 1; k <= ncfft/2; k++ {
		fpk := tmp[k]
		fpnk := cpx{tmp[ncfft-k].r, -tmp[ncfft-k].i}
		fixdiv(&fpk, 2)
		fixdiv(&fpnk, 2)
		f1k := cpx{fpk.r + fpnk.r, fpk.i + fpnk.i}
		f2k := cpx{fpk.r - fpnk.r, fpk.i - fpnk.i}
		tw := cmul(f2k, st.superTwiddles[k-1])
		freq[k].r = int16((int32(f1k.r) + int32(tw.r)) >> 1)
		freq[k].i = int16((int32(f1k.i) + int32(tw.i)) >> 1)
		freq[ncfft-k].r = int16((int32(f1k.r) - int32(tw.r)) >> 1)
		freq[ncfft-k].i = int16((int32(tw.i) - int32(f1k.i)) >> 1)
	}
}

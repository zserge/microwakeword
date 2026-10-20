// Package audiofrontend implements the audio feature extraction pipeline
// used by microWakeWord models. It converts 16 kHz 16-bit PCM audio into
// mel-spectrogram features bit-exact with TensorFlow Lite Micro's
// microfrontend (tensorflow/lite/experimental/microfrontend/lib), as built by
// pymicro-features and ESPHome with microWakeWord's settings.
//
// The pipeline is the reference fixed-point one: Hann window (Q12), input
// scaling to the window's peak, 16-bit KISS FFT, mel filterbank with integer
// square root, noise reduction, PCAN gain control and integer log scaling.
package audiofrontend

import (
	"math"
	"math/bits"
)

// Config holds the frontend configuration parameters.
type Config struct {
	SampleRate   int
	WindowSizeMs int
	StepSizeMs   int
	NumChannels  int
	LowerFreq    float64
	UpperFreq    float64
}

// DefaultConfig returns the standard MWW frontend configuration.
func DefaultConfig(stepMs int) Config {
	return Config{
		SampleRate:   16000,
		WindowSizeMs: 30,
		StepSizeMs:   stepMs,
		NumChannels:  40,
		LowerFreq:    125.0,
		UpperFreq:    7500.0,
	}
}

// Fixed-point constants of the reference implementation.
const (
	windowBits         = 12 // kFrontendWindowBits
	filterbankBits     = 12 // kFilterbankBits
	noiseReductionBits = 14 // kNoiseReductionBits
	pcanSnrBits        = 12 // kPcanSnrBits
	pcanOutputBits     = 6  // kPcanOutputBits
	wideDynamicBits    = 32 // kWideDynamicFunctionBits
	wideDynamicLUTSize = 4*wideDynamicBits - 3

	filterbankIndexAlignment = 4 / 2 // kFilterbankIndexAlignment / sizeof(int16_t)
	filterbankChannelBlock   = 4     // kFilterbankChannelBlockSize

	logScaleLog2    = 16 // kLogScaleLog2
	logSegmentsLog2 = 7  // kLogSegmentsLog2
	logScale        = 65536
	logCoeff        = 45426

	// microWakeWord's settings (pymicro-features init_cfg / ESPHome).
	smoothingBits = 10
	pcanGainBits  = 21
	logScaleShift = 6
)

// Float config values of the reference (C float); vars so that products with
// them are rounded at run time in float32 like the C code.
var (
	evenSmoothing      = float32(0.025)
	oddSmoothing       = float32(0.06)
	minSignalRemaining = float32(0.05)
	pcanStrength       = float32(0.95)
	pcanOffset         = float32(80.0)
)

// Frontend processes audio samples into features. Noise-reduction estimates
// carry over between ProcessFrame calls, as in the reference's stream.
type Frontend struct {
	cfg        Config
	windowSize int
	stepSize   int
	fftSize    int

	// window
	coefficients []int16
	windowed     []int16

	// fft
	fft     *kissFFTR
	fftIn   []int16
	fftOut  []cpx
	energy  []int32
	sqrtBuf []uint32

	// filterbank
	startIndex, endIndex int
	chanFreqStarts       []int16
	chanWeightStarts     []int16
	chanWidths           []int16
	weights, unweights   []int16
	work                 []uint64

	// noise reduction
	evenSmoothing, oddSmoothing, minSignalRemaining uint16
	estimate                                        []uint32

	// pcan
	gainLUT  []int16
	snrShift int32

	correctionBits int
}

// New creates a new Frontend with the given config.
func New(cfg Config) *Frontend {
	windowSize := cfg.WindowSizeMs * cfg.SampleRate / 1000
	stepSize := cfg.StepSizeMs * cfg.SampleRate / 1000
	fftSize := 1
	for fftSize < windowSize {
		fftSize <<= 1
	}
	f := &Frontend{
		cfg:        cfg,
		windowSize: windowSize,
		stepSize:   stepSize,
		fftSize:    fftSize,
		windowed:   make([]int16, windowSize),
		fft:        newKissFFTR(fftSize),
		fftIn:      make([]int16, fftSize),
		fftOut:     make([]cpx, fftSize/2+1),
		// Padded: aligned filterbank channels may read (with zero weight) a
		// few bins past the spectrum, as the C code reads past its buffer.
		energy:   make([]int32, fftSize/2+1+filterbankChannelBlock),
		sqrtBuf:  make([]uint32, cfg.NumChannels),
		estimate: make([]uint32, cfg.NumChannels),
	}
	f.populateWindow()
	f.populateFilterbank(fftSize/2 + 1)

	// NoiseReductionPopulateState: float config * (1 << 14), truncated to uint16.
	f.evenSmoothing = uint16(evenSmoothing * (1 << noiseReductionBits))
	f.oddSmoothing = uint16(oddSmoothing * (1 << noiseReductionBits))
	f.minSignalRemaining = uint16(minSignalRemaining * (1 << noiseReductionBits))

	f.correctionBits = bits.Len32(uint32(fftSize)) - 1 - filterbankBits/2
	f.populatePcan(int32(f.correctionBits))
	return f
}

// Reset restores the initial (zero) noise estimates.
func (f *Frontend) Reset() {
	clear(f.estimate)
}

// ProcessFrame extracts features from one window of samples (windowSize
// samples; missing samples count as zeros). Successive calls must be given
// windows advanced by stepSize samples, matching the reference's stream.
func (f *Frontend) ProcessFrame(samples []int16) []uint16 {
	// WindowProcessSamples
	var maxAbs int16
	for i, c := range f.coefficients {
		var s int16
		if i < len(samples) {
			s = samples[i]
		}
		v := int16((int32(s) * int32(c)) >> windowBits)
		f.windowed[i] = v
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}

	// FftCompute, scaled up so the fixed-point FFT keeps resolution.
	inputShift := 15 - bits.Len32(uint32(maxAbs))
	for i, v := range f.windowed {
		f.fftIn[i] = int16(uint16(v) << inputShift)
	}
	clear(f.fftIn[len(f.windowed):])
	f.fft.transform(f.fftIn, f.fftOut)

	// FilterbankConvertFftComplexToEnergy
	for i := f.startIndex; i < f.endIndex; i++ {
		re, im := int32(f.fftOut[i].r), int32(f.fftOut[i].i)
		f.energy[i] = re*re + im*im
	}

	// FilterbankAccumulateChannels
	var weightAcc, unweightAcc uint64
	for ch := range f.work {
		mags := f.energy[f.chanFreqStarts[ch]:]
		ws := int(f.chanWeightStarts[ch])
		width := int(f.chanWidths[ch])
		weights := f.weights[ws : ws+width]
		unweights := f.unweights[ws : ws+width]
		for j := range width {
			m := uint64(int64(mags[j]))
			weightAcc += uint64(int64(weights[j])) * m
			unweightAcc += uint64(int64(unweights[j])) * m
		}
		f.work[ch] = weightAcc
		weightAcc = unweightAcc
		unweightAcc = 0
	}

	// FilterbankSqrt
	signal := f.sqrtBuf
	for i := range signal {
		signal[i] = sqrt64(f.work[i+1]) >> inputShift
	}

	f.noiseReduce(signal)
	f.pcanApply(signal)

	// LogScaleApply
	out := make([]uint16, len(signal))
	for i, value := range signal {
		if f.correctionBits < 0 {
			value >>= -f.correctionBits
		} else {
			value <<= f.correctionBits
		}
		if value > 1 {
			value = logScaled(value, logScaleShift)
		} else {
			value = 0
		}
		if value > 0xFFFF {
			value = 0xFFFF
		}
		out[i] = uint16(value)
	}
	return out
}

// ProcessAllInt8 processes the entire audio buffer and returns int8 features.
func (f *Frontend) ProcessAllInt8(samples []int16) [][]int8 {
	f.Reset()
	var frames [][]int8
	for offset := 0; offset+f.windowSize <= len(samples); offset += f.stepSize {
		raw := f.ProcessFrame(samples[offset : offset+f.windowSize])
		frame := make([]int8, len(raw))
		for i, v := range raw {
			val := (int32(v)*256+333)/666 - 128
			frame[i] = int8(max(-128, min(127, val)))
		}
		frames = append(frames, frame)
	}
	return frames
}

// populateWindow is WindowPopulateState: a Hann window in float32, rounded
// to Q12.
func (f *Frontend) populateWindow() {
	f.coefficients = make([]int16, f.windowSize)
	arg := float32(math.Pi) * 2.0 / float32(f.windowSize)
	for i := range f.coefficients {
		c := float32(math.Cos(float64(float32(arg * (float32(i) + 0.5)))))
		v := float32(0.5 - float32(0.5*c))
		f.coefficients[i] = int16(math.Floor(float64(float32(v*(1<<windowBits)) + 0.5)))
	}
}

// freqToMel rounds its product explicitly: callers add to and subtract from
// the result, and arm64 would otherwise fuse the multiply into that.
func freqToMel(freq float32) float32 {
	return float32(1127.0 * log1pf(freq/700.0))
}

// populateFilterbank is FilterbankPopulateState, float32 throughout.
func (f *Frontend) populateFilterbank(spectrumSize int) {
	numChannels := f.cfg.NumChannels
	n1 := numChannels + 1
	lower := float32(f.cfg.LowerFreq)
	upper := float32(f.cfg.UpperFreq)

	f.chanFreqStarts = make([]int16, n1)
	f.chanWeightStarts = make([]int16, n1)
	f.chanWidths = make([]int16, n1)
	f.work = make([]uint64, n1)
	actualStarts := make([]int16, n1)
	actualWidths := make([]int16, n1)

	// CalculateCenterFrequencies(num_channels + 1, ...)
	centers := make([]float32, n1)
	melLow := freqToMel(lower)
	melHi := freqToMel(upper)
	melSpacing := (melHi - melLow) / float32(n1)
	for i := range centers {
		centers[i] = melLow + float32(melSpacing*float32(i+1))
	}

	hzPerSbin := float32(0.5*float32(f.cfg.SampleRate)) / (float32(spectrumSize) - 1)
	f.startIndex = int(1.5 + float32(lower/hzPerSbin))
	f.endIndex = 0

	chanFreqIndexStart := f.startIndex
	weightIndexStart := 0
	needsZeros := false
	for ch := range n1 {
		freqIndex := chanFreqIndexStart
		for freqToMel(float32(float32(freqIndex)*hzPerSbin)) <= centers[ch] {
			freqIndex++
		}
		width := freqIndex - chanFreqIndexStart
		actualStarts[ch] = int16(chanFreqIndexStart)
		actualWidths[ch] = int16(width)
		if width == 0 {
			f.chanFreqStarts[ch] = 0
			f.chanWeightStarts[ch] = 0
			f.chanWidths[ch] = filterbankChannelBlock
			if !needsZeros {
				needsZeros = true
				for j := range ch {
					f.chanWeightStarts[j] += filterbankChannelBlock
				}
				weightIndexStart += filterbankChannelBlock
			}
		} else {
			alignedStart := (chanFreqIndexStart / filterbankIndexAlignment) * filterbankIndexAlignment
			alignedWidth := chanFreqIndexStart - alignedStart + width
			paddedWidth := ((alignedWidth-1)/filterbankChannelBlock + 1) * filterbankChannelBlock
			f.chanFreqStarts[ch] = int16(alignedStart)
			f.chanWeightStarts[ch] = int16(weightIndexStart)
			f.chanWidths[ch] = int16(paddedWidth)
			weightIndexStart += paddedWidth
		}
		chanFreqIndexStart = freqIndex
	}

	f.weights = make([]int16, weightIndexStart)
	f.unweights = make([]int16, weightIndexStart)
	for ch := range n1 {
		frequency := int(actualStarts[ch])
		numFrequencies := int(actualWidths[ch])
		frequencyOffset := frequency - int(f.chanFreqStarts[ch])
		weightStart := int(f.chanWeightStarts[ch])
		denom := melLow
		if ch > 0 {
			denom = centers[ch-1]
		}
		for j := 0; j < numFrequencies; j, frequency = j+1, frequency+1 {
			w := (centers[ch] - freqToMel(float32(float32(frequency)*hzPerSbin))) / (centers[ch] - denom)
			idx := weightStart + frequencyOffset + j
			f.weights[idx] = int16(math.Floor(float64(float32(w*(1<<filterbankBits)) + 0.5)))
			f.unweights[idx] = int16(math.Floor(float64(float32((1.0-w)*(1<<filterbankBits)) + 0.5)))
		}
		if frequency > f.endIndex {
			f.endIndex = frequency
		}
	}
	if f.endIndex >= spectrumSize {
		panic("audiofrontend: filterbank end index is above spectrum size")
	}
}

// pcanGainLookup is PcanGainLookupFunction (float32).
func pcanGainLookup(inputBits int32, x uint32) int16 {
	xf := float32(x) / float32(uint32(1)<<inputBits)
	// Rounded explicitly, like the C float variable, so the +0.5 below is not
	// fused into the multiply on arm64.
	gain := float32(float32(uint32(1)<<pcanGainBits) *
		float32(math.Pow(float64(xf+pcanOffset), float64(-pcanStrength))))
	if gain > 0x7FFF {
		return 0x7FFF
	}
	return int16(gain + 0.5)
}

// populatePcan is PcanGainControlPopulateState. The LUT is stored without
// the C code's -6 pointer offset: entry 4*interval+k lives at 4*interval-6+k.
func (f *Frontend) populatePcan(inputCorrectionBits int32) {
	f.snrShift = pcanGainBits - inputCorrectionBits - pcanSnrBits
	inputBits := int32(smoothingBits) - inputCorrectionBits
	lut := make([]int16, wideDynamicLUTSize)
	lut[0] = pcanGainLookup(inputBits, 0)
	lut[1] = pcanGainLookup(inputBits, 1)
	for interval := 2; interval <= wideDynamicBits; interval++ {
		x0 := uint32(1) << (interval - 1)
		x1 := x0 + (x0 >> 1)
		x2 := 2 * x0
		if interval == wideDynamicBits {
			x2 = x0 + (x0 - 1)
		}
		y0 := pcanGainLookup(inputBits, x0)
		y1 := pcanGainLookup(inputBits, x1)
		y2 := pcanGainLookup(inputBits, x2)
		diff1 := int32(y1) - int32(y0)
		diff2 := int32(y2) - int32(y0)
		a1 := 4*diff1 - diff2
		a2 := diff2 - a1
		base := 4*interval - 6
		lut[base] = y0
		lut[base+1] = int16(a1)
		lut[base+2] = int16(a2)
	}
	f.gainLUT = lut
}

// noiseReduce is NoiseReductionApply.
func (f *Frontend) noiseReduce(signal []uint32) {
	for i := range signal {
		smoothing := uint32(f.evenSmoothing)
		if i&1 != 0 {
			smoothing = uint32(f.oddSmoothing)
		}
		oneMinus := uint32(1<<noiseReductionBits) - smoothing
		scaledUp := signal[i] << smoothingBits
		estimate := uint32((uint64(scaledUp)*uint64(smoothing) +
			uint64(f.estimate[i])*uint64(oneMinus)) >> noiseReductionBits)
		f.estimate[i] = estimate
		if estimate > scaledUp {
			estimate = scaledUp
		}
		floor := uint32((uint64(signal[i]) * uint64(f.minSignalRemaining)) >> noiseReductionBits)
		subtracted := (scaledUp - estimate) >> smoothingBits
		signal[i] = max(subtracted, floor)
	}
}

// pcanApply is PcanGainControlApply.
func (f *Frontend) pcanApply(signal []uint32) {
	for i := range signal {
		gain := uint32(int32(wideDynamicFunction(f.estimate[i], f.gainLUT)))
		snr := uint32((uint64(signal[i]) * uint64(gain)) >> f.snrShift)
		signal[i] = pcanShrink(snr)
	}
}

func wideDynamicFunction(x uint32, lut []int16) int16 {
	if x <= 2 {
		return lut[x]
	}
	interval := int16(bits.Len32(x))
	l := lut[4*int(interval)-6:]
	var fr uint32
	if interval < 11 {
		fr = x << (11 - interval)
	} else {
		fr = x >> (interval - 11)
	}
	frac := int32(int16(fr & 0x3FF))
	result := (int32(l[2]) * frac) >> 5
	result += int32(uint32(int32(l[1])) << 5)
	result *= frac
	result = (result + (1 << 14)) >> 15
	result += int32(l[0])
	return int16(result)
}

func pcanShrink(x uint32) uint32 {
	if x < (2 << pcanSnrBits) {
		return (x * x) >> (2 + 2*pcanSnrBits - pcanOutputBits)
	}
	return (x >> (pcanSnrBits - pcanOutputBits)) - (1 << pcanOutputBits)
}

func sqrt32(num uint32) uint16 {
	if num == 0 {
		return 0
	}
	var res uint32
	maxBit := (32 - bits.Len32(num)) | 1
	bit := uint32(1) << (31 - maxBit)
	for iterations := (31-maxBit)/2 + 1; iterations > 0; iterations-- {
		if num >= res+bit {
			num -= res + bit
			res = (res >> 1) + bit
		} else {
			res >>= 1
		}
		bit >>= 2
	}
	if num > res && res != 0xFFFF {
		res++
	}
	return uint16(res)
}

func sqrt64(num uint64) uint32 {
	// The reference takes the 32-bit path whenever the upper word is clear.
	if num>>32 == 0 {
		return uint32(sqrt32(uint32(num)))
	}
	var res uint64
	maxBit := (64 - bits.Len64(num)) | 1
	bit := uint64(1) << (63 - maxBit)
	for iterations := (63-maxBit)/2 + 1; iterations > 0; iterations-- {
		if num >= res+bit {
			num -= res + bit
			res = (res >> 1) + bit
		} else {
			res >>= 1
		}
		bit >>= 2
	}
	if num > res && res != 0xFFFFFFFF {
		res++
	}
	return uint32(res)
}

// logScaled is Log() from log_scale.c: an integer natural log scaled by
// 2^scaleShift.
func logScaled(x uint32, scaleShift uint32) uint32 {
	integer := uint32(bits.Len32(x)) - 1
	fraction := log2FractionPart(x, integer)
	log2 := (integer << logScaleLog2) + fraction
	const round = logScale / 2
	loge := uint32((uint64(logCoeff)*uint64(log2) + round) >> logScaleLog2)
	return ((loge << scaleShift) + round) >> logScaleLog2
}

func log2FractionPart(x, log2x uint32) uint32 {
	frac := int32(int64(x) - int64(1)<<log2x)
	if log2x < logScaleLog2 {
		frac <<= logScaleLog2 - log2x
	} else {
		frac >>= log2x - logScaleLog2
	}
	baseSeg := uint32(frac >> (logScaleLog2 - logSegmentsLog2))
	const segUnit = (uint32(1) << logScaleLog2) >> logSegmentsLog2
	c0 := int32(logLUT[baseSeg])
	c1 := int32(logLUT[baseSeg+1])
	segBase := int32(segUnit * baseSeg)
	relPos := ((c1 - c0) * (frac - segBase)) >> logScaleLog2
	return uint32(frac + c0 + relPos)
}

// logLUT is kLogLut from log_lut.c.
var logLUT = [...]uint16{
	0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163,
	2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
	3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
	5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
	5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
	5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
	5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
	4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
	3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
	2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
	1094, 963, 830, 695, 559, 421, 282, 142, 0, 0,
}

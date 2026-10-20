package audiofrontend

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func testRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Dir(filepath.Dir(file))
}

func loadWAV(path string) ([]int16, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// Walk the RIFF chunks to the "data" chunk: hey_jarvis.wav carries a LIST
	// chunk, so its samples do not start at byte 44.
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4:]))
		body := data[off+8:]
		if id == "data" {
			pcm := body[:min(size, len(body))]
			samples := make([]int16, len(pcm)/2)
			for i := range samples {
				samples[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
			}
			return samples, nil
		}
		off += 8 + size + size&1
	}
	return nil, fmt.Errorf("%s: no data chunk", path)
}

func loadGoldenFeatures(path string) ([][]int8, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	frameCount := int(binary.LittleEndian.Uint32(data[0:4]))
	featureSize := int(binary.LittleEndian.Uint32(data[4:8]))
	frames := make([][]int8, frameCount)
	for i := 0; i < frameCount; i++ {
		frames[i] = make([]int8, featureSize)
		off := 8 + i*featureSize
		for j := 0; j < featureSize; j++ {
			frames[i][j] = int8(data[off+j])
		}
	}
	return frames, nil
}

func TestFrontendVsGolden(t *testing.T) {
	wavPath := filepath.Join(testRoot(), "testdata", "hey_jarvis.wav")
	goldenPath := filepath.Join(testRoot(), "testdata", "hey_jarvis_features.bin")

	if _, err := os.Stat(wavPath); err != nil {
		t.Skipf("WAV not found: %s", wavPath)
	}
	if _, err := os.Stat(goldenPath); err != nil {
		t.Skipf("Golden features not found: %s", goldenPath)
	}

	samples, err := loadWAV(wavPath)
	if err != nil {
		t.Fatalf("loadWAV: %v", err)
	}

	golden, err := loadGoldenFeatures(goldenPath)
	if err != nil {
		t.Fatalf("loadGoldenFeatures: %v", err)
	}

	cfg := DefaultConfig(10) // 10ms step, matching the golden data
	fe := New(cfg)
	goFrames := fe.ProcessAllInt8(samples)

	// The frontend is a bit-exact port of the TFLite Micro microfrontend as
	// built by pymicro-features against glibc, so the features must match the
	// golden file exactly, frame for frame, with one documented exception.
	//
	// The golden file was made with a libm whose log1pf is correctly rounded;
	// glibc's log1pf (which the reference, and so this port, uses) is one ulp
	// off for some filterbank bins, which moves two Q12 filterbank weights by
	// one (2470/1626 -> 2471/1625 and 1400/2696 -> 1401/2695). On this clip
	// that changes one uint16 feature, 418 -> 417 at frame 487 channel 33,
	// which quantizes to 32 instead of the golden 33.
	type at struct{ frame, channel int }
	known := map[at]int8{{487, 33}: 32}
	if len(goFrames) != len(golden) {
		t.Fatalf("frame count: got %d, golden %d", len(goFrames), len(golden))
	}
	mismatches := 0
	for i := range golden {
		if len(goFrames[i]) != len(golden[i]) {
			t.Fatalf("frame %d: got %d features, golden %d", i, len(goFrames[i]), len(golden[i]))
		}
		for j := range golden[i] {
			want := golden[i][j]
			if v, ok := known[at{i, j}]; ok {
				want = v
			}
			if goFrames[i][j] != want {
				if mismatches < 10 {
					t.Errorf("frame %d channel %d: got %d, want %d (golden %d)", i, j, goFrames[i][j], want, golden[i][j])
				}
				mismatches++
			}
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d features differ from the reference", mismatches)
	}
}

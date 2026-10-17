package interp

import (
	"fmt"
	"math"
	"testing"
)

// Stats accumulates the agreement with the oracle over a family of cases.
//
// A relative or ULP error is meaningless for an entry that sits next to a
// zero crossing: the oracle's own rounding error is then larger than the
// entry. So an entry counts for the ULP and relative columns only when it is
// at least 2^-10 of the largest magnitude in its vector ("well-conditioned");
// the Scaled column covers every entry: |got-want| / max|want| of the vector.
type Stats struct {
	N, Exact, Fail int
	WellN          int
	MaxULP         uint64
	MaxRel         float64
	MaxScaled      float64
	Worst, WorstULP string
}

// cmp compares got with want element-wise. Both NaN, or the same value, count
// as exact; NaN or an infinity against a different value is a failure.
func (s *Stats) cmp(t testing.TB, name string, got, want []float64) {
	t.Helper()
	s.cmpF(t, name, got, want, 0)
}

// cmpF is cmp with a floor under the scale: an entry whose true value is
// exactly zero (a third derivative of a quadratic, say) is then judged
// against the size of the problem, not against its own rounding noise.
func (s *Stats) cmpF(t testing.TB, name string, got, want []float64, floor float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: length %d, want %d", name, len(got), len(want))
		s.Fail++
		return
	}
	scale := floor
	for _, w := range want {
		if !math.IsNaN(w) && !math.IsInf(w, 0) {
			scale = math.Max(scale, math.Abs(w))
		}
	}
	for i := range got {
		g, w := got[i], want[i]
		s.N++
		switch {
		case g == w || (math.IsNaN(g) && math.IsNaN(w)):
			s.Exact++
		case math.IsNaN(g) || math.IsNaN(w) || math.IsInf(g, 0) || math.IsInf(w, 0):
			s.Fail++
			t.Errorf("%s[%d]: got %v, want %v", name, i, g, w)
		default:
			sc := math.Abs(g-w) / scale
			if sc > s.MaxScaled {
				s.MaxScaled, s.Worst = sc, fmt.Sprintf("%s[%d]", name, i)
			}
			if math.Abs(w) >= scale/1024 {
				s.WellN++
				if u := ulpDiff(g, w); u > s.MaxULP {
					s.MaxULP, s.WorstULP = u, fmt.Sprintf("%s[%d]", name, i)
				}
				s.MaxRel = math.Max(s.MaxRel, math.Abs(g-w)/math.Abs(w))
			}
		}
	}
}

func (s *Stats) cmpScaled(t testing.TB, name string, got, want []float64) {
	t.Helper()
	s.cmp(t, name, got, want)
}

func (s *Stats) report(t testing.TB, family string) {
	t.Helper()
	t.Logf("%-24s values %6d exact %6d  well-conditioned %6d maxULP %5d (%s) maxRel %.2e  maxScaled %.2e (%s)",
		family, s.N, s.Exact, s.WellN, s.MaxULP, s.WorstULP, s.MaxRel, s.MaxScaled, s.Worst)
}

// within fails the test if the family exceeds its budget.
func (s *Stats) within(t testing.TB, family string, maxULP uint64, maxScaled float64) {
	t.Helper()
	if s.Fail > 0 {
		t.Errorf("%s: %d failing entries", family, s.Fail)
	}
	if s.MaxULP > maxULP {
		t.Errorf("%s: max ULP %d exceeds budget %d", family, s.MaxULP, maxULP)
	}
	if s.MaxScaled > maxScaled {
		t.Errorf("%s: max scaled error %.3g exceeds budget %.3g (%s)", family, s.MaxScaled, maxScaled, s.Worst)
	}
}

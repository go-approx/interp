package interp

import "math"

// gtsv solves the tridiagonal system with sub-diagonal dl (n-1), diagonal d
// (n) and super-diagonal du (n-1) for one right-hand side b, in place, by
// Gaussian elimination with partial pivoting; it is LAPACK's dgtsv, which
// scipy.linalg.solve_banded calls for the (1, 1) band. dl, d, du and b are
// overwritten. It reports false if the matrix is exactly singular.
func gtsv(dl, d, du, b []float64) bool {
	n := len(d)
	for i := 0; i < n-2; i++ {
		if math.Abs(d[i]) >= math.Abs(dl[i]) {
			if d[i] == 0 {
				return false
			}
			fact := dl[i] / d[i]
			d[i+1] -= float64(fact * du[i])
			b[i+1] -= float64(fact * b[i])
			dl[i] = 0
		} else {
			fact := d[i] / dl[i]
			d[i] = dl[i]
			temp := d[i+1]
			d[i+1] = du[i] - float64(fact*temp)
			dl[i] = du[i+1]
			du[i+1] = -float64(fact * dl[i])
			du[i] = temp
			temp = b[i]
			b[i] = b[i+1]
			b[i+1] = temp - float64(fact*b[i+1])
		}
	}
	if n > 1 {
		i := n - 2
		if math.Abs(d[i]) >= math.Abs(dl[i]) {
			if d[i] == 0 {
				return false
			}
			fact := dl[i] / d[i]
			d[i+1] -= float64(fact * du[i])
			b[i+1] -= float64(fact * b[i])
		} else {
			fact := d[i] / dl[i]
			d[i] = dl[i]
			temp := d[i+1]
			d[i+1] = du[i] - float64(fact*temp)
			du[i] = temp
			temp = b[i]
			b[i] = b[i+1]
			b[i+1] = temp - float64(fact*b[i+1])
		}
	}
	if d[n-1] == 0 {
		return false
	}
	b[n-1] /= d[n-1]
	if n > 1 {
		b[n-2] = (b[n-2] - float64(du[n-2]*b[n-1])) / d[n-2]
	}
	for i := n - 3; i >= 0; i-- {
		b[i] = (b[i] - float64(du[i]*b[i+1]) - float64(dl[i]*b[i+2])) / d[i]
	}
	return true
}

// denseSolve solves the n-by-n system a x = b (a row-major, overwritten; b
// overwritten with x) by LU with partial pivoting. It reports false if a
// zero pivot is met.
func denseSolve(a []float64, b []float64, n int) bool {
	for j := 0; j < n; j++ {
		p := j
		for i := j + 1; i < n; i++ {
			if math.Abs(a[i*n+j]) > math.Abs(a[p*n+j]) {
				p = i
			}
		}
		if a[p*n+j] == 0 {
			return false
		}
		if p != j {
			for c := 0; c < n; c++ {
				a[j*n+c], a[p*n+c] = a[p*n+c], a[j*n+c]
			}
			b[j], b[p] = b[p], b[j]
		}
		r := 1 / a[j*n+j]
		for i := j + 1; i < n; i++ {
			l := a[i*n+j] * r
			for c := j + 1; c < n; c++ {
				a[i*n+c] -= float64(l * a[j*n+c])
			}
			b[i] -= float64(l * b[j])
		}
	}
	for j := n - 1; j >= 0; j-- {
		b[j] /= a[j*n+j]
		for i := 0; i < j; i++ {
			b[i] -= float64(b[j] * a[i*n+j])
		}
	}
	return true
}

// bandLU is the LU factorization with partial pivoting of a square band
// matrix with kl sub-diagonals and ku super-diagonals (LAPACK dgbtrf). Row i
// stores columns i-kl .. i+ku+kl, the extra kl columns being fill-in.
type bandLU struct {
	n, kl, ku int
	w         int
	a         []float64 // n*w, row-major band
	lm        []float64 // n*kl multipliers: lm[j*kl+(i-j-1)] for row i, column j
	piv       []int
}

func newBand(n, kl, ku int) *bandLU {
	w := 2*kl + ku + 1
	return &bandLU{n: n, kl: kl, ku: ku, w: w, a: make([]float64, n*w), lm: make([]float64, n*kl), piv: make([]int, n)}
}

// at returns the storage index of element (i, j), which must lie in the band.
func (m *bandLU) at(i, j int) int { return i*m.w + (j - i + m.kl) }

// set stores a(i, j) = v; (i, j) must be inside the band kl/ku.
func (m *bandLU) set(i, j int, v float64) { m.a[m.at(i, j)] = v }

// factor computes the LU factorization; it reports false if the matrix is
// singular (an exactly zero pivot).
func (m *bandLU) factor() bool {
	n, kl, ku := m.n, m.kl, m.ku
	for j := 0; j < n; j++ {
		last := min(j+kl, n-1)
		p := j
		for i := j + 1; i <= last; i++ {
			if math.Abs(m.a[m.at(i, j)]) > math.Abs(m.a[m.at(p, j)]) {
				p = i
			}
		}
		m.piv[j] = p
		if m.a[m.at(p, j)] == 0 {
			return false
		}
		cend := min(j+ku+kl, n-1)
		if p != j {
			for c := j; c <= cend; c++ {
				ip, ij := m.at(p, c), m.at(j, c)
				m.a[ip], m.a[ij] = m.a[ij], m.a[ip]
			}
		}
		r := 1 / m.a[m.at(j, j)]
		for i := j + 1; i <= last; i++ {
			l := m.a[m.at(i, j)] * r
			m.lm[j*kl+(i-j-1)] = l
			for c := j + 1; c <= cend; c++ {
				m.a[m.at(i, c)] -= float64(l * m.a[m.at(j, c)])
			}
		}
	}
	return true
}

// solve overwrites b (length n) with the solution of A x = b.
func (m *bandLU) solve(b []float64) {
	n, kl, ku := m.n, m.kl, m.ku
	for j := 0; j < n; j++ {
		if p := m.piv[j]; p != j {
			b[j], b[p] = b[p], b[j]
		}
		last := min(j+kl, n-1)
		for i := j + 1; i <= last; i++ {
			b[i] -= float64(b[j] * m.lm[j*kl+(i-j-1)])
		}
	}
	for j := n - 1; j >= 0; j-- {
		if b[j] != 0 {
			b[j] /= m.a[m.at(j, j)]
			t := b[j]
			for i := j - 1; i >= max(0, j-(kl+ku)); i-- {
				b[i] -= float64(t * m.a[m.at(i, j)])
			}
		}
	}
}

// failure is the panic value of must; guard turns it back into an error at
// the public entry points, so that internal solves that cannot fail for valid
// input do not clutter every call with an error path.
type failure struct{ err error }

// must panics with ErrSingular if ok is false.
func must(ok bool) {
	if !ok {
		panic(failure{ErrSingular})
	}
}

// guard is deferred by the entry points that call must; it stores the
// panic's error in *err.
func guard(err *error) {
	if r := recover(); r != nil {
		*err = r.(failure).err
	}
}

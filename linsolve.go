package interp

import "math"

// gtsv solves the tridiagonal system with sub-diagonal dl (n-1), diagonal d
// (n) and super-diagonal du (n-1) for one right-hand side b, in place, by
// Gaussian elimination with partial pivoting; it is LAPACK's dgtsv, which
// scipy.linalg.solve_banded calls for the (1, 1) band. dl, d, du and b are
// overwritten. It reports false if the matrix is exactly singular.
func gtsv(dl, d, du, b []float64) bool {
	n := len(d)
	// Fast path: while no row interchange is called for — always, for the
	// diagonally dominant systems of the splines — eliminate with the same
	// operations as the general loop below, without its stores (dl[i] = 0).
	// The first row that needs an interchange hands over to the general
	// loop, which then sees exactly the state it would have built itself.
	clean := 0
	for clean < n-1 && d[clean] != 0 && math.Abs(d[clean]) >= math.Abs(dl[clean]) {
		i := clean
		fact := dl[i] / d[i]
		d[i+1] -= float64(fact * du[i])
		b[i+1] -= float64(fact * b[i])
		clean++
	}
	if clean == n-1 {
		return gtsvBack(dl, d, du, b, clean)
	}
	for k := 0; k < min(clean, n-2); k++ {
		dl[k] = 0 // what the general loop would have stored for these rows
	}
	for i := clean; i < n-2; i++ {
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
	if n > 1 && clean <= n-2 {
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
	return gtsvBack(dl, d, du, b, 0)
}

// gtsvBack is dgtsv's back substitution. Rows below clean were eliminated by
// the fast path, whose dl entries are logically zero (not stored): their
// term is computed as 0*b[i+2], as the general path computes dl[i]*b[i+2]
// with the dl[i] = 0 it stored, so signed zeros and NaNs come out the same.
func gtsvBack(dl, d, du, b []float64, clean int) bool {
	n := len(d)
	if d[n-1] == 0 {
		return false
	}
	b[n-1] /= d[n-1]
	if n > 1 {
		b[n-2] = (b[n-2] - float64(du[n-2]*b[n-1])) / d[n-2]
	}
	i := n - 3
	for ; i >= clean; i-- {
		b[i] = (b[i] - float64(du[i]*b[i+1]) - float64(dl[i]*b[i+2])) / d[i]
	}
	var zero float64
	for ; i >= 0; i-- {
		b[i] = (b[i] - float64(du[i]*b[i+1]) - float64(zero*b[i+2])) / d[i]
	}
	return true
}

// gtsv2 is gtsv for two right-hand sides b and c sharing one factorization:
// the same pivots and the same operations, in the same order, as two gtsv
// calls on copies of (dl, d, du), so each solution is bit-identical to
// gtsv's.
func gtsv2(dl, d, du, b, c []float64) bool {
	n := len(d)
	for i := 0; i < n-1; i++ {
		if math.Abs(d[i]) >= math.Abs(dl[i]) {
			if d[i] == 0 {
				return false
			}
			fact := dl[i] / d[i]
			d[i+1] -= float64(fact * du[i])
			b[i+1] -= float64(fact * b[i])
			c[i+1] -= float64(fact * c[i])
			if i < n-2 {
				dl[i] = 0
			}
		} else {
			fact := d[i] / dl[i]
			d[i] = dl[i]
			temp := d[i+1]
			d[i+1] = du[i] - float64(fact*temp)
			if i < n-2 {
				dl[i] = du[i+1]
				du[i+1] = -float64(fact * dl[i])
			}
			du[i] = temp
			temp = b[i]
			b[i] = b[i+1]
			b[i+1] = temp - float64(fact*b[i+1])
			temp = c[i]
			c[i] = c[i+1]
			c[i+1] = temp - float64(fact*c[i+1])
		}
	}
	if d[n-1] == 0 {
		return false
	}
	b[n-1] /= d[n-1]
	c[n-1] /= d[n-1]
	if n > 1 {
		b[n-2] = (b[n-2] - float64(du[n-2]*b[n-1])) / d[n-2]
		c[n-2] = (c[n-2] - float64(du[n-2]*c[n-1])) / d[n-2]
	}
	for i := n - 3; i >= 0; i-- {
		b[i] = (b[i] - float64(du[i]*b[i+1]) - float64(dl[i]*b[i+2])) / d[i]
		c[i] = (c[i] - float64(du[i]*c[i+1]) - float64(dl[i]*c[i+2])) / d[i]
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
	n, kl, ku, w := m.n, m.kl, m.ku, m.w
	for j := 0; j < n; j++ {
		last := min(j+kl, n-1)
		rj := m.a[j*w : (j+1)*w] // row j: column c is at c-j+kl
		p := j
		best := math.Abs(rj[kl])
		for i := j + 1; i <= last; i++ {
			if v := math.Abs(m.a[i*w+j-i+kl]); v > best {
				p, best = i, v
			}
		}
		m.piv[j] = p
		if m.a[p*w+j-p+kl] == 0 {
			return false
		}
		cend := min(j+ku+kl, n-1)
		if p != j {
			rp := m.a[p*w : (p+1)*w]
			for c := j; c <= cend; c++ {
				rp[c-p+kl], rj[c-j+kl] = rj[c-j+kl], rp[c-p+kl]
			}
		}
		r := 1 / rj[kl]
		for i := j + 1; i <= last; i++ {
			ri := m.a[i*w : (i+1)*w]
			l := ri[j-i+kl] * r
			m.lm[j*kl+(i-j-1)] = l
			if l == 0 {
				continue
			}
			for off := 1; off <= cend-j; off++ {
				ri[j+off-i+kl] -= float64(l * rj[j+off-j+kl])
			}
		}
	}
	return true
}

// solve overwrites b (length n) with the solution of A x = b.
func (m *bandLU) solve(b []float64) {
	n, kl, ku, w := m.n, m.kl, m.ku, m.w
	for j := 0; j < n; j++ {
		if p := m.piv[j]; p != j {
			b[j], b[p] = b[p], b[j]
		}
		last := min(j+kl, n-1)
		lm := m.lm[j*kl : (j+1)*kl]
		bj := b[j]
		for i := j + 1; i <= last; i++ {
			b[i] -= float64(bj * lm[i-j-1])
		}
	}
	for j := n - 1; j >= 0; j-- {
		if b[j] != 0 {
			b[j] /= m.a[j*w+kl]
			t := b[j]
			for i := j - 1; i >= max(0, j-(kl+ku)); i-- {
				b[i] -= float64(t * m.a[i*w+j-i+kl])
			}
		}
	}
}

// solveRows solves A X = B for s right-hand sides at once, B stored as the
// n rows of c (row i is c[i*s:(i+1)*s]), overwriting c with X. Every column
// goes through exactly the operations solve applies to it, in the same
// order (zero entries skipped in the back substitution as solve skips
// them), so each column of X is bit-identical to solve's result; walking
// the rows contiguously is what makes it faster than gathering columns.
func (m *bandLU) solveRows(c []float64, s int) {
	n, kl, ku, w := m.n, m.kl, m.ku, m.w
	for j := 0; j < n; j++ {
		rj := c[j*s : (j+1)*s : (j+1)*s]
		if p := m.piv[j]; p != j {
			rp := c[p*s : (p+1)*s : (p+1)*s]
			for k := range rj {
				rj[k], rp[k] = rp[k], rj[k]
			}
		}
		last := min(j+kl, n-1)
		lm := m.lm[j*kl : (j+1)*kl]
		for i := j + 1; i <= last; i++ {
			l := lm[i-j-1]
			ri := c[i*s : (i+1)*s : (i+1)*s]
			for k, v := range rj {
				ri[k] -= float64(v * l)
			}
		}
	}
	col := make([]float64, kl+ku) // a[i][j] for the rows i the back substitution updates
	rows := make([][]float64, kl+ku)
	for j := n - 1; j >= 0; j-- {
		d := m.a[j*w+kl]
		lo := max(0, j-(kl+ku))
		nu := j - lo
		for q := 0; q < nu; q++ {
			i := j - 1 - q
			col[q] = m.a[i*w+j-i+kl]
			rows[q] = c[i*s : (i+1)*s : (i+1)*s]
		}
		rj := c[j*s : (j+1)*s : (j+1)*s]
		for k, v := range rj {
			if v == 0 {
				continue
			}
			v /= d
			rj[k] = v
			for q := 0; q < nu; q++ {
				rows[q][k] -= float64(v * col[q])
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

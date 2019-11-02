#include <Rcpp.h>
#include <stdint.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]

/*
 * See http://xoroshiro.di.unimi.it/ for more information about the
 * random number generator
 */

// Protect against compilers without OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline double to_double(uint64_t x) {
  const union { uint64_t i; double d; } u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
  return u.d - 1.0;
}

/* This is a fixed-increment version of Java 8's SplittableRandom generator
 See http://dx.doi.org/10.1145/2714064.2660195 and
 http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html
 */

uint64_t split_mix_64(uint64_t x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

uint64_t next(uint64_t &t1, uint64_t &t2) {
  const uint64_t s0 = t1;
  uint64_t s1 = t2;
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  t1 = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  t2 = rotl(s1, 36); // c

  return result;
}

/* This is the jump function for the generator. It is equivalent
 to 2^64 calls to next(); it can be used to generate 2^64
 non-overlapping subsequences for parallel computations. */
void fast_rand_jump(uint64_t &t1, uint64_t &t2) {
  static const uint64_t j[] = { 0xbeac0467eba5facb, 0xd86b048b86aa9922 };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  for(uint64_t i = 0; i < sizeof j / sizeof j[0]; i++)
    for(int b = 0; b < 64; b++) {
      if ((j[i] & UINT64_C(1) << b) != 0) {
        s0 ^= t1;
        s1 ^= t2;
      }
      next(t1, t2);
    }

    t1 = s0;
  t2 = s1;
}


void set_fast_seed(int n, uint64_t &t1, uint64_t &t2) {
  // Use the split_mix_64 generator to seed the first batch
  t1 = split_mix_64(n);
  t2 = split_mix_64(t1);
}

/*
// [[Rcpp::export(rng = false)]]
NumericVector fast_rand(int n, int stream = 0){
  NumericVector out(n);

  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  set_fast_seed(435346, s1, s2);

  for(int i = 0; i < stream; ++i){
    fast_rand_jump(s1, s2);
  }

  for(int i = 0; i < n; ++i) {
    out[i] = to_double(next(s1, s2));
  }
  return out;
}
*/

static inline double
  ws (std::vector<float> cw, 
      uint64_t &s1, uint64_t &s2){
    double u = to_double(next(s1, s2));
    for (std::size_t i = 0, e = cw.size(); i != e; ++i){
      if (u < cw[i]) return(i);
    }
    return(0);
  }

/*
// [[Rcpp::export(rng = false)]]
NumericVector fast_ws(int n, 
                      NumericVector cw,
                      int N,
                      int stream = 0){
  NumericVector out(n);
  
  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  set_fast_seed(435346, s1, s2);
  
  for(int i = 0; i < stream; ++i){
    fast_rand_jump(s1, s2);
  }
  
  for(int i = 0; i < n; ++i) {
    out[i] = ws(cw, s1, s2);
  }
  return out;
}
*/

static inline int 
  cpois (double lambda, uint64_t &s1, uint64_t &s2){
    int k = 0;
    double L = exp(-lambda);
    double p = 1.0;
    while (p > L){
      p = p*to_double(next(s1, s2));
      k++;
    }
    return(k - 1);
  }

/*
// [[Rcpp::export(rng = false)]]
NumericVector fast_pois(int n, float lambda, int stream = 0){
  NumericVector out(n);
  
  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  set_fast_seed(435346, s1, s2);
  
  for(int i = 0; i < stream; ++i){
    fast_rand_jump(s1, s2);
  }
  
  for(int i = 0; i < n; ++i) {
    out[i] = cpois(lambda, s1, s2);
  }
  return out;
}
 */

static inline float
  fastlog2 (float x)
  {
    union { float f; uint32_t i; } vx = { x };
    union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;

    return y - 124.22551499f
    - 1.498030302f * mx.f
      - 1.72587999f / (0.3520887068f + mx.f);
  }

static inline float
  fastlog (float x)
  {
    return 0.69314718f * fastlog2 (x);
  }

// [[Rcpp::export(rng = false)]]
void run_sim_cpp(int iters, const int M, const int N,
                 const std::vector<int> index,
                 const std::vector<float> U,
                 const std::vector<float> EU,
                 const std::vector<float> V,
                 const std::vector<float> s_v,
                 const std::vector<float> obs_util,
                 const std::vector<float> alt_obs_util,
                 const std::vector<float> h,
                 const std::vector<float> a,
                 const std::vector<float> b,
                 const std::vector<float> sw_cv,
                 const std::vector<float> sw_ev,
                 const std::vector<float> fc,
                 const std::vector<float> inflator,
                 const std::vector<float> sub_disp,
                 const std::vector<float> sub_alt_disp,
                 const std::vector<int> ID,
                 NumericMatrix out,
                 NumericMatrix CV,
                 NumericMatrix EV,
                 NumericMatrix disp_matrix,
                 const std::vector<float> cw,
                 const std::vector<float> lambda,
                 int ncores){

  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  set_fast_seed(432635754, s1, s2);

  uint64_t s[N][2];

  auto max_id = *max_element(std::begin(ID), std::end(ID));
  std::set<int> id_set(ID.begin(), ID.end());

  int count = 0;
  for (int i = 1; i < max_id; ++i){
    fast_rand_jump(s1, s2);
    if(id_set.find(i) != id_set.end()){
      s[count][0] = s1;
      s[count][1] = s2;
      count++;
    }
  }

  // Declare local variables
  float err_obs = 0;
  float log_err_obs = 0;
  float max_u = 0;
  float max_ind = 0;
  float err = 0;
  float u = 0;

  // float c = 0;
  // float int_cv = 0;
  // float cv = 0;
  // float min_cv = 0;
  // float sum_cv = 0;
  // float start_cv = 0;

  // float int_ev = 0;
  // float ev = 0;
  // float min_ev = 0;
  // float sum_ev = 0;
  // float start_ev = 0;

  float obs_diff = 0;
  
  std::vector<float> cwi(1000, 0.0);
  std::vector<float> ref_dist(1000, 0.0);
  float lam = 0;
  float s_pois = 0;
  int j = 0;

// #pragma omp parallel for num_threads(ncores) firstprivate(err_obs, max_u, max_ind, u, log_err_obs, err, c, int_cv, cv, min_cv, sum_cv, start_cv, int_ev, ev, min_ev, sum_ev, start_ev, cwi, lam, j)
#pragma omp parallel for num_threads(ncores) firstprivate(err_obs, max_u, max_ind, u, log_err_obs, err, cwi, lam, j, ref_dist, s_pois)
  for(int i = 0; i < N; ++i) {

    // Derive the indices to check in this case
    obs_diff = V[index[i]-1] - U[index[i]-1];

    // Reset variables
    // sum_cv = 0;
    // sum_ev = 0;

    // Calculate welfare at the observed point
    // c = (-obs_util[i] + h[index[i]-1]);
    // int_cv = -b[index[i]-1]/(2*a[i]) + sqrt(b[index[i]-1]*b[index[i]-1] - 4*a[i]*c)/(2*a[i]*sw_cv[index[i]-1]);
    // start_cv = (int_cv + fc[index[i]-1])*5200*inflator[i] - sub_alt_disp[index[i]-1];

    // c = (-alt_obs_util[i] + h[index[i]-1]);
    // int_ev = -b[index[i]-1]/(2*a[i]) + sqrt(b[index[i]-1]*b[index[i]-1] - 4*a[i]*c)/(2*a[i]*sw_ev[index[i]-1]);
    // start_ev = sub_disp[index[i]-1] - (int_ev + fc[index[i]-1])*5200*inflator[i];

    // Extract the cumulative probabilties and lambda
    for (int j = 0; j < M - 1; ++j){
      cwi[j] = cw[i + (j+1)*N];
    }
    
    for (int it = 0; it < iters; ++it){
      
      // Need to determine sampled utility and s_v here
      // Sample the number of choices from the Poisson
      lam = cpois(lambda[i], s[i][0], s[i][1]);
      // Sample from the NN reference distribution
      s_pois = 1;
      for (int k = 0; k < lam; ++k){
        ref_dist[k] = 1 + ws(cwi, s[i][0], s[i][1]);
        s_pois += exp(U[i + ref_dist[k]]);
      }
      s_pois = 1/s_pois;

      // Calculate the error at the observed point
      err_obs = -s_pois*fastlog((float) to_double(next(s[i][0], s[i][1])));
      log_err_obs = -fastlog(err_obs);
      max_u = V[index[i]-1] + log_err_obs;
      max_ind = index[i]-1;

      // min_cv = start_cv;
      // min_ev = start_ev;

      // Calculate the error at the remaining points
      for (int k = 0; k < lam; ++k){
        
        // Get index j for this iteration - FIX
        j = ref_dist[k];

        if (V[i + N*j] - U[i + N*j] <= obs_diff) continue;

        err = - fastlog(- fastlog((float) to_double(next(s[i][0], s[i][1]))) + EU[i + N*j] * err_obs);
        u = V[i + N*j] + err;

        if (V[i + N*j] - U[i + N*j] > obs_diff && u > max_u){
          max_u = u;
          max_ind = i + N*j;
        }

        /*
        // Welfare calculation
        err = err + fastlog(err_obs);
        c = (-obs_util[i] + err + h[i + N*j]);

        if (b[i + N*j]*b[i + N*j] - 4*a[i]*c > 0){
          int_cv = -b[i + N*j]/(2*a[i]) + sqrt(b[i + N*j]*b[i + N*j] - 4*a[i]*c)/(2*a[i]*sw_cv[i + N*j]);
          cv = (int_cv + fc[i + N*j])*5200*inflator[i] - sub_alt_disp[i + N*j];
          if (cv < min_cv){
            min_cv = cv;
          }
        }

        c = (-alt_obs_util[i] + err + h[i + N*j]);

        if (b[i + N*j]*b[i + N*j] - 4*a[i]*c > 0){
          int_ev = -b[i + N*j]/(2*a[i]) + sqrt(b[i + N*j]*b[i + N*j] - 4*a[i]*c)/(2*a[i]*sw_ev[i + N*j]);
          ev = sub_disp[i + N*j] - (int_ev + fc[i + N*j])*5200*inflator[i];
          if (ev > min_ev){
            min_ev = ev;
          }
        }
        */

      }

      // Update the output
      out[max_ind] += 1;
      // sum_cv += min_cv;
      // sum_ev += min_ev;
      // Save the disposable income for this scenario
      disp_matrix[i + it*N] = sub_alt_disp[max_ind];
    }

    // CV[i] = sum_cv/iters;
    // EV[i] = sum_ev/iters;
  }

}



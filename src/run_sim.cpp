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

static inline double
  ws (std::vector<float> cw, 
      uint64_t &s1, uint64_t &s2){
    double u = to_double(next(s1, s2));
    for (std::size_t i = 0, e = cw.size(); i != e; ++i){
      if (u < cw[i]) return(i);
    }
    return(0);
  }

static inline int 
  cpois (const double lambda, uint64_t &s1, uint64_t &s2){
    int k = 0;
    double L = exp(-lambda);
    double p = 1.0;
    while (p > L){
      p = p*to_double(next(s1, s2));
      k++;
    }
    return(k - 1);
  }

// [[Rcpp::export(rng = false)]]
void run_sim_cpp_parallel(const int iters, 
                 const int M, const int N,
                 const std::vector<int> in_matrix,
                 const std::vector<float> U,
                 const std::vector<float> V,
                 const std::vector<int> ID,
                 NumericMatrix out_matrix,
                 const std::vector<float> cw,
                 const std::vector<float> lambda,
                 const int ncores, const int seed){

  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  set_fast_seed(seed, s1, s2);

  uint64_t s[2*N];

  auto max_id = *max_element(std::begin(ID), std::end(ID));
  std::set<int> id_set(ID.begin(), ID.end());

  int count = 0;
  for (int i = 1; i <= max_id; ++i){
    fast_rand_jump(s1, s2);
    if(id_set.find(i) != id_set.end()){
      s[2*count] = s1;
      s[2*count + 1] = s2;
      count++;
    }
  }

  const int max_choices = 100;
  std::vector<float> ui(M);
  std::vector<float> vi(M);

#pragma omp parallel for num_threads(ncores) firstprivate(ui, vi) default(none) shared(s, out_matrix)
  for(int i = 0; i < N; ++i) {
    
    // Extract the utility so we can mess with it later
    for (int j = 0; j < M; ++j){
      ui[j] = U[i + j*N];
      vi[j] = V[i + j*N];
    }
    
    std::vector<float> cwi(M - 1);
    std::vector<int> ref_dist(max_choices);
    
    for (int it = 0; it < iters; ++it){
      
      int index = in_matrix[i + N*it];
      float du = ui[index];
      
      for (int j = 0; j < M; ++j){
        ui[j] -= du;
        vi[j] -= du;
      }
      
      float obs_diff = vi[index] - ui[index];
      
      // Extract the cumulative probabilities and lambda
      
      /* 
       * The next for loop is easily the nastiest section of the entire simulations.
       * I ended up rewriting from scratch in Julia as a cross-check and
       * found some issues.  The latest version seems like it works...
       */
      float diff_prob = 0;
      if (index > 0){
        diff_prob = cw[i + (index - 1)*N] - 
          (index > 1 ? cw[i + (index - 2)*N] : 0);
      }

      int mod_j = 0;
      for (int j = 0; j < M; ++j){
        if (j == 0 && j != index){
          // Hot-swap the probability of the replacement option
          cwi[0] = diff_prob;
          continue;
        }
        if (j == index) continue;
        mod_j = (j < index) ? j : (j - 1);
        cwi[mod_j] = cw[i + (j - 1)*N] + 
          ((j < index) ? diff_prob : 0);
      }
      
      // Need to determine sampled utility and s_v here
      // Sample the number of choices from the Poisson
      // Limit to max choices to avoid computational issues
      int lam = cpois(lambda[i], s[2*i], s[2*i + 1]);
      lam = (lam > max_choices) ? max_choices : lam;
      // Sample from the NN reference distribution
      float s_pois = 1;
      int exc_count = 0;
      int inc_count = 0;
      int test = 0;
      
      for (int k = 0; k < lam; ++k){
        
        test = ws(cwi, s[2*i], s[2*i + 1]);
        test += ((ref_dist[inc_count] >= index) ? 1 : 0);
        
        if (vi[test] - ui[test] <= obs_diff){
          exc_count++;
          continue;
        }
        
        ref_dist[inc_count] = test;
        inc_count++;
        
        s_pois += exp(ui[ref_dist[inc_count]]);
      }
      lam -= exc_count;
      s_pois = 1.0/s_pois;
      
      // Calculate the error at the observed point
      float err_obs = -s_pois*log(to_double(next(s[2*i], s[2*i + 1])));
      float max_u = vi[index] - log(err_obs);
      int max_ind = index;
      
      // Calculate the error at the remaining points
      float err = 0;
      for (int k = 0; k < lam; ++k){

        // Get index for this iteration
        int choice = ref_dist[k];
        
        err = - log(- log(to_double(next(s[2*i], s[2*i + 1]))) + exp(ui[choice])* err_obs);
        
        if (vi[choice] + err > max_u){
          max_u = vi[choice] + err;
          max_ind = choice;
        }
        
      }
      
      // Save the results (by reference)
      out_matrix[i + it*N] = max_ind;
    }
    
  }
}



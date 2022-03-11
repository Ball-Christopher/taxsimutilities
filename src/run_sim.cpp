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


static inline double to_double(uint64_t x){
  return (x >> 11) * 0x1.0p-53;
				   
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

uint64_t next(uint64_t &t1, uint64_t &t2,
              uint64_t &t3, uint64_t &t4) {
  const uint64_t result = rotl(t1 + t4, 23) + t1;
  const uint64_t t = t2 << 17;
  
  t3 ^= t1;
  t4 ^= t2;
  t2 ^= t3;
  t1 ^= t4;
  
  t3 ^= t;
  
  t4 = rotl(t4, 45);
  
  return result;
}

/* This is the jump function for the generator. It is equivalent
 to 2^64 calls to next(); it can be used to generate 2^64
 non-overlapping subsequences for parallel computations. */
void fast_rand_jump(uint64_t &t1, uint64_t &t2,
                    uint64_t &t3, uint64_t &t4) {
  static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
  
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for(unsigned int i = 0; i < 4; i++){
    for(int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= t1;
        s1 ^= t2;
        s2 ^= t3;
        s3 ^= t4;
      }
      next(t1, t2, t3, t4);	
    }
  }
  
  t1 = s0;
  t2 = s1;
  t3 = s2;
  t4 = s3;
}


void set_fast_seed(int n, uint64_t &t1, uint64_t &t2,
                   uint64_t &t3, uint64_t &t4) {
  // Use the split_mix_64 generator to seed the first batch
  t1 = split_mix_64(n);
  t2 = split_mix_64(t1);
  t3 = split_mix_64(t2);
  t4 = split_mix_64(t3);
}

static inline double
  ws (std::vector<double> &cw,
      uint64_t &s1, uint64_t &s2,
      uint64_t &s3, uint64_t &s4){
    // Note: changing from cw to &cw helps.  A lot.
    // Could potentially be faster with binary search...
    // or the Walker algorithm...
    double u = to_double(next(s1, s2, s3, s4));
    int i = 0;
    // This assumes that the last value of cw is 1 (or more).
    while (cw[i] > u) i++;
    return(i);
  }


static inline int 
  cpois (const double &exp_neg_lambda, uint64_t &s1, uint64_t &s2,
         uint64_t &s3, uint64_t &s4){
    // This is the fastest method I can find for lambda < 30.
    int k = 0;
							
    double p = 1.0;
    while (p > exp_neg_lambda){
      p = p*to_double(next(s1, s2, s3, s4));
      k++;
    }
    return(k - 1);
  }

//' @export
// [[Rcpp::export(rng = false)]]
void run_sim_cpp_parallel(const int iters, 
                          const int M, const int N,
                          const std::vector<int> in_matrix,
                          const std::vector<double> U,
                          const std::vector<double> V,
                          const std::vector<int> ID,
                          NumericMatrix out_matrix,
                          const std::vector<double> cw,
                          const std::vector<double> lambda,
                          const int ncores, const int seed){
  
  // Set up the random number generator
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  uint64_t s4 = 0;
  set_fast_seed(seed, s1, s2, s3, s4);
  
  uint64_t s[4*N];
  
  auto max_id = *max_element(std::begin(ID), std::end(ID));
  std::set<int> id_set(ID.begin(), ID.end());
  
  int count = 0;
  for (int i = 1; i <= max_id; ++i){
						   
    if(id_set.find(i) != id_set.end()){
      s[4*count] = s1;
      s[4*count + 1] = s2;
      s[4*count + 2] = s3;
      s[4*count + 3] = s4;
      count++;
    }
    fast_rand_jump(s1, s2, s3, s4);
  }
  
  const int max_choices = 100;
  std::vector<double> ui(M);
  std::vector<double> eui(M);
  std::vector<double> vi(M);
  
  uint64_t x1 = 0;
  uint64_t x2 = 0;
  uint64_t x3 = 0;
  uint64_t x4 = 0;
  
  std::vector<double> cwi(M - 1);
  std::vector<int> ref_dist(max_choices);
  double exp_neg_lambda = 0.0;
  double err = 0.0;
  int choice = 0;
  double err_obs = 0.0;
  double max_u = 0.0;
  int max_ind = 0;
  int lam = 0;
  int index = 0;
  double du = 0.0;
  double obs_diff = 0.0;
  double diff_prob = 0.0;
  int mod_j = 0;
  double s_pois = 1.0;
  int exc_count = 0;
  int inc_count = 0;
  int test = 0;
  
#pragma omp parallel for num_threads(ncores) firstprivate(ui, vi, eui, x1, x2, x3, x4, cwi, ref_dist, exp_neg_lambda, err, choice, err_obs, max_u, max_ind, lam, index, du, obs_diff, diff_prob, mod_j, s_pois, exc_count, inc_count, test) default(none) shared(s, out_matrix)
  for(int i = 0; i < N; ++i) {
    
    x1 = s[4*i];
    x2 = s[4*i + 1];
    x3 = s[4*i + 2];
    x4 = s[4*i + 3];
    
    // Extract the utility so we can mess with it later
    for (int j = 0; j < M; ++j){
      ui[j] = U[i + j*N];
      vi[j] = V[i + j*N];
      eui[j] = exp(ui[j]);
    }
    
    exp_neg_lambda = exp(-lambda[i]);
										   
    
    for (int it = 0; it < iters; ++it){
      
      index = in_matrix[i + N*it];
      du = ui[index];
      
      for (int j = 0; j < M; ++j){
        ui[j] -= du;
        vi[j] -= du;
      }
      
      obs_diff = vi[index] - ui[index];
      
      // Extract the cumulative probabilities and lambda
      
      /* 
       * The next for loop is easily the nastiest section of the entire simulations.
       * I ended up rewriting from scratch in Julia as a cross-check and
       * found some issues.  The latest version seems like it works...
       */
      diff_prob = 0.0;
      if (index > 0) {
        diff_prob = cw[i + (index - 1)*N] - 
          (index > 1 ? cw[i + (index - 2)*N] : 0.0);
      }
      
      mod_j = 0;
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
      // Sample the number of choices from the Poisson.
      // Limit to max choices to avoid computational issues
      lam = cpois(exp_neg_lambda, x1, x2, x3, x4);
      lam = (lam > max_choices) ? max_choices : lam;
      // Sample from the NN reference distribution
      s_pois = 1.0;
      exc_count = 0;
      inc_count = 0;
      test = 0;
      
      for (int k = 0; k < lam; ++k){
        
        test = ws(cwi, x1, x2, x3, x4);
        test += ((ref_dist[inc_count] >= index) ? 1 : 0);
        
        if (vi[test] - ui[test] <= obs_diff){
          exc_count++;
          continue;
        }
        
        ref_dist[inc_count] = test;
        inc_count++;
        
        // Weird error here with the previous code.  Using 
        // exp(ui[ref_dist[inc_count]]) instead of exp(ui[test])
        // convert to an integer, which messes with the distribution
        // of the observed error term.  
        s_pois += eui[test];
      }
      lam -= exc_count;
      s_pois = 1.0/s_pois;
      
      // Calculate the error at the observed point
      // double uni = to_double(next(x1, x2, x3, x4));
      err_obs = -s_pois*log(to_double(next(x1, x2, x3, x4)));
      max_u = vi[index] - log(err_obs);
      max_ind = index;
      
      // Calculate the error at the remaining points
					
      for (int k = 0; k < lam; ++k){
        
        // Get index for this iteration
        choice = ref_dist[k];
        
        err = - log(- log(to_double(next(x1, x2, x3, x4))) + 
          eui[choice]*err_obs);
        
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




#include <Rcpp.h>
#include <math.h> 
using namespace Rcpp;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]

// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]

// Protect against compilers without OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::export]]
double gini_sorted(const std::vector<double> y,
                   const std::vector<double> w) {
  double gini = 0;
  double totw = 0;
  for (auto& i : w) totw += i;
  double totnu = 0;
  for (unsigned int i=0; i < w.size(); i++) {
    totnu += w[i]*y[i];
  }
  totnu /= totw;
  double cumw = 0;
  double cumnu = 0;
  double wn = w[0]/totw;
  double nun = wn*y[0]/totnu;
  for (unsigned int i=0; i < y.size() - 1; ++i){
    cumw += wn;
    cumnu += nun;
    wn = w[i + 1]/totw;
    nun = wn*y[i + 1]/totnu;
    gini += (cumnu + nun)*cumw - cumnu*(cumw + wn);
  }
  return gini;
}

// [[Rcpp::export]]
NumericMatrix fast_poverty(const std::vector<double> y,
                           const std::vector<double> w,
                           const std::vector<double> k,
                           int ncores) {
  
  unsigned int N = y.size();
  int rows = w.size()/N;
  NumericMatrix out( rows , 24 );
  
  // Can we speed this up with multi-core???
#pragma omp parallel for num_threads(ncores)
  for (unsigned int col=0; col < w.size(); col += N){
    
    int ocol = col/N;
    // Calculate the median - assumes sorted
    double totw = 0;
    for (unsigned int i=0; i < N; i++) {
      totw += w[col + i];
    }
    
    double totwk = 0;
    for (unsigned int i=0; i < N; i++) {
      totwk += w[col + i]*k[i];
    }
    
    double cumw = 0;
    for (unsigned int i=0; i < N; ++i){
      cumw += w[col+i];
      if (cumw > totw/2){
        out(ocol, 0) = ((cumw - totw/2)/w[col+i])*y[i-1] + (1.0 - (cumw - totw/2)/w[col+i])*y[i];
        break;
      }
    }
    
    // Calculate income poverty
    double th50 = 0.5*out(ocol, 0);
    double th60 = 0.6*out(ocol, 0);
    cumw = 0;
    double cumwk = 0;
    double cumy50 = 0;
    double cumy502 = 0;
    double cumy60 = 0;
    double cumy602 = 0;
    double watts50 = 0;
    double watts60 = 0;
    bool fin50 = false;
    for (unsigned int i=0; i < N; ++i){
      if (y[i] > th50 && !fin50){
        out(ocol, 1) = cumw/totw;
        out(ocol, 2) = cumwk/totwk;
        out(ocol, 3) = cumy50/totw;
        out(ocol, 20) = cumy50/cumw;
        out(ocol, 4) = cumy502/totw;
        out(ocol, 16) = watts50/totw;
        fin50 = true;
      }
      if (y[i] > th60){
        out(ocol, 5) = cumw/totw;
        out(ocol, 6)= cumwk/totwk;
        out(ocol, 7) = cumy60/totw;
        out(ocol, 21) = cumy60/cumw;
        out(ocol, 8) = cumy602/totw;
        out(ocol, 17) = watts60/totw;
        break;
      }
      cumw += w[col+i];
      cumwk += k[i]*w[col+i];
      cumy50 += w[col+i]*(th50 - y[i])/th50;
      cumy502 += w[col+i]*(th50 - y[i])*(th50 - y[i])/(th50*th50);
      cumy60 += w[col+i]*(th60 - y[i])/th60;
      cumy602 += w[col+i]*(th60 - y[i])*(th60 - y[i])/(th60*th60);
      watts50 += w[col+i]*log(th50/y[i]);
      watts60 += w[col+i]*log(th60/y[i]);
    }
    
    // Now for the inequality measures
    double gini = 0;
    double totnu = 0;
    for (unsigned int i=0; i < N; i++) {
      totnu += w[col + i]*y[i];
    }
    totnu /= totw;
    cumw = 0;
    double cumnu = 0;
    double wn = w[col]/totw;
    double nun = wn*y[0]/totnu;
    for (unsigned int i=0; i < N - 1; ++i){
      cumw += wn;
      cumnu += nun;
      wn = w[col + i + 1]/totw;
      nun = wn*y[i + 1]/totnu;
      gini += (cumnu + nun)*cumw - cumnu*(cumw + wn);
    }
    out(ocol, 9) = gini;
    
    // Theil
    double ybar = totnu;
    double lybar = log(ybar);
    std::vector<double> logy(N);
    for (unsigned int i=0; i < N; i++) {
      logy[i] = log(y[i]);
    }
    double l = totw*lybar;
    for (unsigned int i=0; i < N; ++i){
      l -= w[col+i]*logy[i];
    }
    out(ocol, 10) = 1.0/totw*l;
    
    double t = 0;
    for (unsigned int i=0; i < N; ++i){
      t += w[col+i]*y[i]*(logy[i] - lybar);
    }
    t = 1.0/ybar*t;
    out(ocol, 11) = 1.0/totw*t;
    
    std::vector<double> yscaled(N);
    for (unsigned int i=0; i < N; i++) {
      yscaled[i] = y[i]/ybar;
    }
    double ge = 0;
    for (unsigned int i=0; i < N; ++i){
      ge += w[col+i]*yscaled[i]*yscaled[i];
    }
    ge = 1/totw*ge - 1;
    out(ocol, 12) = 1.0/2*ge;
    
    // Atkinson
    double atk05 = 0;
    double atk1 = 1;
    double atk2 = 0;
    for (unsigned int i=0; i < N; ++i){
      atk05 += w[col+i]*sqrt(yscaled[i]);
      atk1 *= pow(y[i], w[i]/totw);
      atk2 += w[col+i]*1/yscaled[i];
    }
    out(ocol, 13) = 1.0 - atk05/totw*atk05/totw;
    out(ocol, 14) = 1.0 - atk1/ybar;
    out(ocol, 15) = totw/atk2;
    
    // Let's tack on the Sen-Shorrock-Thon index for completeness
    std::vector<double> pgap50(N); 
    std::vector<double> pgap60(N); 
    std::vector<double> rw(N); 
    for (unsigned int i=0; i < N; ++i){
      rw[N - 1 - i] = w[col+i];
      if (y[i] < th50){
        pgap50[N - 1 - i] = (th50 - y[i])/th50;
      }
      if (y[i] < th60){
        pgap60[N - 1 - i] = (th60 - y[i])/th60;
      }
    }
    
    // Export the SST poverty measures
    out(ocol, 22) = gini_sorted(pgap50, rw);
    out(ocol, 23) = gini_sorted(pgap60, rw);
    out(ocol, 18) = out(ocol, 1)*out(ocol, 20)*(1.0 + out(ocol, 22));
    out(ocol, 19) = out(ocol, 5)*out(ocol, 21)*(1.0 + out(ocol, 23));
  }
  return out;
}

// [[Rcpp::export]]
double fast_med(NumericVector y,
                NumericVector w) {
  double tot = sum(w);
  double cumw = 0;
  for (int i=0; i < y.size(); ++i){
    cumw += w[i];
    if (cumw > tot/2){
      return ((cumw - tot/2)/w[i])*y[i-1] + (1.0 - (cumw - tot/2)/w[i])*y[i];
    }
  }
  return -1;
}

// [[Rcpp::export]]
double fast_pov(NumericVector y,
                NumericVector w,
                double thres) {
  double th = thres*fast_med(y, w);
  double cumw = 0;
  for (int i=0; i < y.size(); ++i){
    if (y[i] > th){
      return cumw;
    }
    cumw += w[i];
  }
  return -1;
}

// [[Rcpp::export]]
double fast_cpov(NumericVector y,
                 NumericVector w,
                 NumericVector k,
                 double thres) {
  double th = thres*fast_med(y, w);
  double cumw = 0;
  for (int i=0; i < y.size(); ++i){
    if (y[i] > th){
      return cumw;
    }
    cumw += k[i]*w[i];
  }
  return -1;
}

// [[Rcpp::export]]
double fast_povgap(NumericVector y,
                   NumericVector w,
                   double thres,
                   double exp) {
  double th = thres*fast_med(y, w);
  double cumw = 0;
  double cumy = 0;
  for (int i=0; i < y.size(); ++i){
    if (y[i] > th){
      return cumy/cumw;
    }
    cumw += w[i];
    if (exp == 1) cumy += w[i]*(th - y[i])/th;
    else cumy += w[i]*pow(1 - y[i]/th, exp);
  }
  return -1;
}

// [[Rcpp::export]]
double theil_l(NumericVector y,
               NumericVector w) {
  double totw = sum(w);
  double ybar = sum(y*w)/totw;
  double l = totw*log(ybar);
  for (int i=0; i < y.size(); ++i){
    l -= w[i]*log(y[i]);
  }
  return 1.0/totw*l;
}

// [[Rcpp::export]]
double theil_t(NumericVector y,
               NumericVector w) {
  double totw = sum(w);
  double ybar = sum(w*y)/totw;
  double lybar = log(ybar);
  double t = 1.0/ybar*sum(w*y*(log(y) - lybar));
  return 1.0/totw*t;
}

// [[Rcpp::export]]
double gen_ent(NumericVector y,
               NumericVector w,
               double a) {
  double N = sum(w);
  double ybar = sum(w*y)/N;
  double ge = 1/N*sum(w*pow(y/ybar, a)) - 1;
  return 1.0/(a*(a-1))*ge;
}

// [[Rcpp::export]]
double atkinson_1(NumericVector y,
                  NumericVector w) {
  double N = sum(w);
  double ybar = sum(w*y)/N;
  double atk = 1;
  for (int i=0; i < y.size(); ++i){
    atk *= pow(y[i], w[i]/N);
  }
  return 1.0 - atk/ybar;
}

// [[Rcpp::export]]
double atkinson_e(NumericVector y,
                  NumericVector w,
                  double e) {
  double N = sum(w);
  double ybar = sum(w*y)/N;
  double atk = 0;
  for (int i=0; i < y.size(); ++i){
    atk += w[i]*pow(y[i]/ybar, 1 - e);
  }
  return 1.0 - pow(atk/N, 1/(1 - e));
}
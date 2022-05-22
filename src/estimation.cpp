#include <Rcpp.h>
#include <math.h>
#include "logSum.hpp"
using namespace Rcpp;

// Added this for older versions of Rcpp
// [[Rcpp::plugins(cpp11)]]

// These are needed somewhere
//' @useDynLib taxsimutilities, .registration = TRUE
//' @importFrom Rcpp evalCpp

//' @export
// [[Rcpp::export]]
double logSumExp (NumericVector& logV, int accumulators=8) {
  // This may be somewhat OS/CPU dependent.
  constexpr int MAX_ACCUMULATORS = 12;
  if (accumulators>MAX_ACCUMULATORS) accumulators = MAX_ACCUMULATORS;  
  return logSumN(&logV[0],logV.size(),accumulators,_int<MAX_ACCUMULATORS>());
}

//' @export
// [[Rcpp::export(rng = false)]]
NumericVector stable_ig(
    const NumericMatrix U,
    const NumericMatrix W,
    const int N, const int M){
  
  NumericVector E(N);
  NumericVector L(N);
  double tmp;
  double D;
  NumericVector ig(N);
  
  for (int j = 1; j < M; ++j){
    for (int i = 0; i < N; ++i){
      tmp = W[i + j*N]*U[i + j*N];
      L[i] += tmp;
      E[i] += tmp*tmp;
    }
  }
  
  for (int i = 0; i < N; ++i){
    L[i] /= (1.0 - W[i]);
    E[i] /= (1.0 - W[i])*(1.0 - W[i]);
    tmp = (E[i] + L[i]*L[i]);
    D = tmp*tmp - 4.0*E[i]*(U[i] - L[i])*(U[i] - L[i]);
    if (D >= 0.0) ig[i] = (tmp + sqrt(D))/(2.0*E[i]);
    else ig[i] = 1.0;
    ig[i] *= U[i]/(U[i] + (ig[i] - 1.0)*L[i]);
  }
  
  return(ig);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
NumericVector stable_n(
    const NumericMatrix U,
    const NumericMatrix W,
    const int N, const int M){
  
  NumericVector E(N);
  NumericVector L(N);
  double tmp;
  double D;
  NumericVector n(N);
  
  for (int j = 1; j < M; ++j){
    for (int i = 0; i < N; ++i){
      tmp = W[i + j*N]*U[i + j*N];
      L[i] += tmp;
      E[i] += tmp*tmp;
    }
  }
  
  for (int i = 0; i < N; ++i){
    L[i] /= (1.0 - W[i]);
    E[i] /= (1.0 - W[i])*(1.0 - W[i]);
    tmp = (E[i] + L[i]*L[i]);
    D = tmp*tmp - 4.0*E[i]*(U[i] - L[i])*(U[i] - L[i]);
    if (D >= 0.0) n[i] = (tmp + sqrt(D))/(2.0*E[i]);
    else n[i] = 1.0;
  }
  
  return(n);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
NumericVector stable_p(
    const NumericMatrix U,
    const NumericMatrix W,
    const int N, const int M){
  
  NumericVector E(N);
  NumericVector L(N);
  double tmp;
  double D;
  NumericVector p(N);
  NumericVector n(N);
  
  for (int j = 1; j < M; ++j){
    for (int i = 0; i < N; ++i){
      tmp = W[i + j*N]*U[i + j*N];
      L[i] += tmp;
      E[i] += tmp*tmp;
    }
  }
  
  for (int i = 0; i < N; ++i){
    L[i] /= (1.0 - W[i]);
    E[i] /= (1.0 - W[i])*(1.0 - W[i]);
    tmp = (E[i] + L[i]*L[i]);
    D = tmp*tmp - 4.0*E[i]*(U[i] - L[i])*(U[i] - L[i]);
    if (D >= 0.0) n[i] = (tmp + sqrt(D))/(2.0*E[i]);
    else n[i] = 1.0;
    p[i] = U[i]/(U[i] + (n[i] - 1.0)*L[i]);
  }
  
  return(p);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double llc_cpp(
    const NumericVector p, 
    const NumericMatrix H1, 
    const NumericMatrix H2, 
    const NumericMatrix H1sq, 
    const NumericMatrix H2sq, 
    const NumericMatrix H1H2, 
    const NumericMatrix Y, 
    const NumericMatrix H1Y, 
    const NumericMatrix H2Y, 
    const NumericMatrix Ysq, 
    const NumericMatrix TW, 
    const NumericVector nw,
    const int &N, const int &M){
  
  NumericMatrix U(N, M);
  std::vector<double> Um(N);
  std::vector<double> d(N);
  // std::vector<double> tmp(M);
  
  double ll = 0;
  
  // Split the utility calculation into two stages to keep
  // it in the cache.  May vary by CPU...
  
  for (int i = 0; i < N*M; i++) {
    U[i] = p[0]*H1[i] + p[1]*H1sq[i] + p[2]*H2[i] +
      p[3]*H2sq[i] + p[4]*H1H2[i];
  }
  
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += p[5]*Y[i + j*N] + p[6]*Ysq[i + j*N] + 
        p[7]*H1Y[i + j*N] + p[8]*H2Y[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  for (int i = 0; i < N*M; i++) {
    d[i%N] += (p[0] + 2*p[1]*H1[i] + 
      p[4]*H2[i] + p[7]*Y[i]) < 0 ? 1.0 : 0.0;
    d[i%N] += (p[2] + 2*p[3]*H2[i] + 
      p[4]*H1[i] + p[8]*Y[i]) < 0 ? 1.0 : 0.0;
    d[i%N] += (p[5] + 2*p[6]*Y[i] + 
      p[7]*H1[i] + p[8]*H2[i]) > 0 ? 1.0 : 0.0;
    U[i] = exp(U[i] - Um[i%N]);
  }
  
  NumericVector ig = stable_ig(U, TW, N, M);
  
  for (int i = 0; i < N; i++) {
    ll += d[i]*nw[i]/M*sqrt(ig[i]);
  }
  
  return(-ll);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double llcopt_cpp(
    const NumericVector p, 
    const NumericMatrix H1, 
    const NumericMatrix H2, 
    const NumericMatrix H1sq, 
    const NumericMatrix H2sq, 
    const NumericMatrix H1H2, 
    const NumericMatrix Y, 
    const NumericMatrix H1Y, 
    const NumericMatrix H2Y, 
    const NumericMatrix Ysq, 
    const NumericMatrix TW, 
    const NumericVector nw,
    const int &N, const int &M, const int &opt_mode){
  
  // Update this - do we need 4 different combinations of positive and negative?
  double phy1 = 0.0;
  double phy2 = 0.0;
  double ph1h2 = 0.0;
  if (opt_mode < 9){
    ph1h2 = p[4];
    phy1 = p[7];
    phy2 = p[8];
  }
  
  if (opt_mode == 9){
    if (p[1]*p[6] < 0.0) return(-p[1]*p[6]);
    if (p[3]*p[6] < 0.0) return(-p[3]*p[6]);
    ph1h2 = p[4];
    phy1 = (p[7] > 0.0 ? 1.0 : -1.0)*sqrt(4*p[1]*p[6]);
    phy2 = (p[8] > 0.0 ? 1.0 : -1.0)*sqrt(4*p[3]*p[6]);
  }
  
  if (opt_mode == 10){
    if (p[1]*p[6] < 0.0) return(-p[1]*p[6]);
    if (p[3]*p[6] < 0.0) return(-p[3]*p[6]);
    ph1h2 = (p[4] > 0.0 ? 1.0 : -1.0)*sqrt(4*p[1]*p[3]);
    phy1 = (p[7] > 0.0 ? 1.0 : -1.0)*sqrt(4*p[1]*p[6]);
    phy2 = (p[8] > 0.0 ? 1.0 : -1.0)*sqrt(4*p[3]*p[6]);
  }
  
  NumericMatrix U(N, M);
  std::vector<double> Um(N);
  std::vector<double> Us(N);
  std::vector<double> d(N);
  
  double ll = 0.0;
  
  for (int i = 0; i < N*M; i++) {
    U[i] = p[0]*H1[i] + p[1]*H1sq[i] + p[2]*H2[i] +
      p[3]*H2sq[i] + ph1h2*H1H2[i];
  }
  
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += p[5]*Y[i + j*N] + p[6]*Ysq[i + j*N] + 
        phy1*H1Y[i + j*N] + phy2*H2Y[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  // Convert to numerically stable version
  for (int i = 0; i < N*M; i++) {
    U[i] = exp(U[i] - Um[i%N]);
  }
  
  if (opt_mode == 1){
    for (int j = 0; j < M; ++j){
      for (int i = 0; i < N; ++i){
        Us[i] += TW[i + j*N]*U[i + j*N];
      }
    }
    for (int i = 0; i < N; i++) {
      ll += nw[i]*log(TW[i]*U[i]/Us[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 3){
    NumericVector ps = stable_p(U, TW, N, M);
    for (int i = 0; i < N; i++) {
      ll += nw[i]*sqrt(M*ps[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 4){
    for (int j = 0; j < M; ++j){
      for (int i = 0; i < N; ++i){
        Us[i] += TW[i + j*N]*U[i + j*N];
      }
    }
    
    std::vector<double> dh1(N);
    std::vector<double> dh2(N);
    std::vector<double> dy(N);
    for (int i = 0; i < N*M; i++) {
      dh1[i%N] += (p[0] + 2.0*p[1]*H1[i] + 
        ph1h2*H2[i] + phy1*Y[i]) < 0.0 ? 1.0 : 0.0;
      dh2[i%N] += (p[2] + 2*p[3]*H2[i] + 
        ph1h2*H1[i] + phy2*Y[i]) < 0.0 ? 1.0 : 0.0;
      dy[i%N] += (p[5] + 2*p[6]*Y[i] + 
        phy1*H1[i] + phy2*H2[i]) > 0.0 ? 1.0 : 0.0;
    }
    
    std::vector<double> std_prob(N);
    double min_prob = 0;
    for (int i = 0; i < N; i++) {
      std_prob[i] = log(TW[i]*U[i]/Us[i]);
      if (std_prob[i] < min_prob) min_prob = std_prob[i];
    }
    
    for (int i = 0; i < N; i++) {
      ll += nw[i]*((dh1[i] < M ? min_prob : std_prob[i]) +
        (dh2[i] < M ? min_prob : std_prob[i]) +
        (dy[i] < M ? min_prob : std_prob[i]));
    }
    return(-ll);
  }
  
  if (opt_mode == 5){
    NumericVector ig = stable_ig(U, TW, N, M);
    for (int i = 0; i < N; i++) {
      ll += nw[i]*sqrt(ig[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 7){
    
    for (int i = 0; i < N*M; i++) {
      d[i%N] += (p[0] + 2.0*p[1]*H1[i] + 
        ph1h2*H2[i] + phy1*Y[i]) < 0 ? 1.0 : 0.0;
      d[i%N] += (p[2] + 2.0*p[3]*H2[i] + 
        ph1h2*H1[i] + phy2*Y[i]) < 0 ? 1.0 : 0.0;
      d[i%N] += (p[5] + 2.0*p[6]*Y[i] + 
        phy1*H1[i] + phy2*H2[i]) > 0 ? 1.0 : 0.0;
    }
    
    NumericVector ps = stable_p(U, TW, N, M);
    
    for (int i = 0; i < N; i++) {
      ll += d[i]/M*nw[i]*sqrt(M*ps[i]);
    }
    return(-ll);
  }
  
  if (opt_mode >= 8){
    NumericVector ig = stable_ig(U, TW, N, M);
    
    for (int i = 0; i < N*M; i++) {
      d[i%N] += (p[0] + 2.0*p[1]*H1[i] + 
        ph1h2*H2[i] + phy1*Y[i]) < 0 ? 1.0 : 0.0;
      d[i%N] += (p[2] + 2.0*p[3]*H2[i] + 
        ph1h2*H1[i] + phy2*Y[i]) < 0 ? 1.0 : 0.0;
      d[i%N] += (p[5] + 2.0*p[6]*Y[i] + 
        phy1*H1[i] + phy2*H2[i]) > 0 ? 1.0 : 0.0;
    }
    
    for (int i = 0; i < N; i++) {
      ll += d[i]/M*nw[i]*sqrt(ig[i]);
    }
    return(-ll);
  }
  
  return(-ll);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double llc_alt_cpp(
    const SEXP p, 
    const SEXP H1, 
    const SEXP H2, 
    const SEXP H1sq, 
    const SEXP H2sq, 
    const SEXP H1H2, 
    const SEXP Y, 
    const SEXP H1Y, 
    const SEXP H2Y, 
    const SEXP Ysq, 
    const SEXP lTW, 
    const SEXP nw,
    const int &N, const int &M){
  
  std::vector<double> U(N*M);
  std::vector<double> Um(N);
  std::vector<double> sumexp(N);
  std::vector<double> tmp(M);
  
  NumericVector pp(p);
  NumericVector w(nw);
  NumericMatrix h1(H1);
  NumericMatrix h1sq(H1sq);
  NumericMatrix h2(H2);
  NumericMatrix h2sq(H2sq);
  NumericMatrix h1y(H1Y);
  NumericMatrix h2y(H2Y);
  NumericMatrix h1h2(H1H2);
  NumericMatrix y(Y);
  NumericMatrix ysq(Ysq);
  NumericMatrix ltw(lTW);
  double ll = 0;
  
  // Split the utility calculation into two stages to keep
  // it in the cache.
  for (int i = 0; i < N*M; i++) {
    U[i] = pp[0]*h1[i] + pp[1]*h1sq[i] + pp[2]*h2[i] +
      pp[3]*h2sq[i] + pp[4]*h1h2[i];
  }
  
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += pp[5]*y[i + j*N] + pp[6]*ysq[i + j*N] + 
        pp[7]*h1y[i + j*N] + pp[8]*h2y[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < M; ++j){
      tmp[j] = ltw[i + j*N] + U[i + j*N] - Um[i];
    }
    sumexp[i] = logSumN(&tmp[0], M, 8, _int<12>());
    ll += w[i]*(U[i] - Um[i] - sumexp[i]);
  }
  
  return(-ll);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double lls_cpp(
    const NumericVector p, 
    const NumericMatrix H,  
    const NumericMatrix Hsq, 
    const NumericMatrix Y, 
    const NumericMatrix Ysq, 
    const NumericMatrix HY,
    const NumericMatrix TW, 
    const NumericVector nw,
    const int &N, const int &M){
  
  NumericMatrix U(N, M);
  std::vector<double> Um(N);
  std::vector<double> d(N);
  // std::vector<double> tmp(M);
  
  double ll = 0;
  
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += p[0]*H[i + j*N] + p[1]*Hsq[i + j*N] +
        p[2]*Y[i + j*N] + p[3]*Ysq[i + j*N] + 
        p[4]*HY[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  for (int i = 0; i < N*M; i++) {
    d[i%N] += (p[0] + 2*p[1]*H[i] + p[4]*Y[i]) < 0 ? 1.0 : 0.0;
    d[i%N] += (p[2] + 2*p[3]*Y[i] + p[4]*H[i]) > 0 ? 1.0 : 0.0;
    U[i] = exp(U[i] - Um[i%N]);
  }
  
  NumericVector ig = stable_ig(U, TW, N, M);
  
  for (int i = 0; i < N; i++) {
    ll += d[i]*nw[i]/M*sqrt(ig[i]);
  }
  
  return(-ll);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double lls_alt_cpp(
    const SEXP p, 
    const SEXP H, 
    const SEXP Hsq, 
    const SEXP Y, 
    const SEXP Ysq, 
    const SEXP HY, 
    const SEXP lTW, 
    const SEXP nw,
    const int &N, const int &M){
  
  std::vector<double> U(N*M);
  std::vector<double> Um(N);
  std::vector<double> sumexp(N);
  std::vector<double> tmp(M);
  
  NumericVector pp(p);
  NumericVector w(nw);
  NumericMatrix h(H);
  NumericMatrix hsq(Hsq);
  NumericMatrix hy(HY);
  NumericMatrix y(Y);
  NumericMatrix ysq(Ysq);
  NumericMatrix ltw(lTW);
  double ll = 0;
  
  // Split the utility calculation into two stages to keep
  // it in the cache.
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += pp[0]*h[i + j*N] + pp[1]*hsq[i + j*N] +
        pp[2]*y[i + j*N] + pp[3]*ysq[i + j*N] + 
        pp[4]*hy[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < M; ++j){
      tmp[j] = ltw[i + j*N] + U[i + j*N] - Um[i];
    }
    sumexp[i] = logSumN(&tmp[0], M, 8, _int<12>());
    ll += w[i]*(U[i] - Um[i] - sumexp[i]);
  }
  
  return(-ll);
  
}

//' @export
// [[Rcpp::export(rng = false)]]
double llsopt_cpp(
    const NumericVector p, 
    const NumericMatrix H,  
    const NumericMatrix Hsq, 
    const NumericMatrix Y, 
    const NumericMatrix Ysq, 
    const NumericMatrix HY,
    const NumericMatrix TW, 
    const NumericVector nw,
    const int &N, const int &M, const int &opt_mode){
  
  double phy = 0;
  if (opt_mode < 9){
    phy = p[4];
  }
  if (opt_mode == 9){
    if (p[1]*p[3] < 0) return(-p[1]*p[3]);
    phy = sqrt(4*p[1]*p[3]);
  }
  
  if (opt_mode == 10){
    if (p[1]*p[3] < 0) return(-p[1]*p[3]);
    phy = -sqrt(p[1]*p[3] > 0);
  }
  
  NumericMatrix U(N, M);
  std::vector<double> Um(N);
  std::vector<double> Us(N);
  std::vector<double> d(N);
  
  double ll = 0;
  
  for (int j = 0; j < M; ++j){
    for (int i = 0; i < N; ++i){
      U[i + j*N] += p[0]*H[i + j*N] + p[1]*Hsq[i + j*N] +
        p[2]*Y[i + j*N] + p[3]*Ysq[i + j*N] + 
        phy*HY[i + j*N];
      if ( (j == 0) || U[i + j*N] > Um[i]) Um[i] = U[i + j*N];
    }
  }
  
  // Convert to numerically stable version
  for (int i = 0; i < N*M; i++) {
    U[i] = exp(U[i] - Um[i%N]);
  }
  
  if (opt_mode == 1){
    for (int j = 0; j < M; ++j){
      for (int i = 0; i < N; ++i){
        Us[i] += TW[i + j*N]*U[i + j*N];
      }
    }
    for (int i = 0; i < N; i++) {
      ll += nw[i]*log(TW[i]*U[i]/Us[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 3){
    NumericVector ps = stable_p(U, TW, N, M);
    for (int i = 0; i < N; i++) {
      ll += nw[i]*sqrt(M*ps[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 4){
    for (int j = 0; j < M; ++j){
      for (int i = 0; i < N; ++i){
        Us[i] += TW[i + j*N]*U[i + j*N];
      }
    }
    
    std::vector<double> dh(N);
    std::vector<double> dy(N);
    for (int i = 0; i < N*M; i++) {
      dh[i%N] += (p[0] + 2*p[1]*H[i] + phy*Y[i]) < 0.0 ? 1.0 : 0.0;
      dy[i%N] += (p[2] + 2*p[3]*Y[i] + phy*H[i]) > 0.0 ? 1.0 : 0.0;
    }
    
    std::vector<double> std_prob(N);
    double min_prob = 0;
    for (int i = 0; i < N; i++) {
      std_prob[i] = log(TW[i]*U[i]/Us[i]);
      if (std_prob[i] < min_prob) min_prob = std_prob[i];
    }
    
    for (int i = 0; i < N; i++) {
      ll += nw[i]*((dh[i] < M ? min_prob : std_prob[i]) + 
        (dy[i] < M ? min_prob : std_prob[i]));
    }
    return(-ll);
  }
  
  if (opt_mode == 5){
    NumericVector ig = stable_ig(U, TW, N, M);
    for (int i = 0; i < N; i++) {
      ll += nw[i]*sqrt(ig[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 7){
    
    for (int i = 0; i < N*M; i++) {
      d[i%N] += (p[0] + 2.0*p[1]*H[i] + phy*Y[i]) < 0.0 ? 1.0 : 0.0;
      d[i%N] += (p[2] + 2.0*p[3]*Y[i] + phy*H[i]) > 0.0 ? 1.0 : 0.0;
    }
    
    NumericVector ps = stable_p(U, TW, N, M);
    
    for (int i = 0; i < N; i++) {
      ll += d[i]/M*nw[i]*sqrt(M*ps[i]);
    }
    return(-ll);
  }
  
  if (opt_mode == 8 || opt_mode == 9 || opt_mode == 10){
    NumericVector ig = stable_ig(U, TW, N, M);
    
    for (int i = 0; i < N*M; i++) {
      d[i%N] += (p[0] + 2*p[1]*H[i] + phy*Y[i]) < 0 ? 1.0 : 0.0;
      d[i%N] += (p[2] + 2*p[3]*Y[i] + phy*H[i]) > 0 ? 1.0 : 0.0;
    }
    
    for (int i = 0; i < N; i++) {
      ll += d[i]/M*nw[i]*sqrt(ig[i]);
    }
    return(-ll);
  }
  
  return(-ll);
  
}